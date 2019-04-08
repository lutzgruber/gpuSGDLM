#include "SGDLM/HostWrapperImpl.cuh"

#include "kernel_functions.cuh"

#include "cublas_manager.cuh"

#include "curand_manager.cuh"

#include "SGDLM/SGDLM.cuh"

template<typename DOUBLE> SGDLM::HostWrapperImpl<DOUBLE>::HostWrapperImpl(std::size_t no_gpus) :
		memory_initialized(false), sim_memory_initialized(false), evo_memory_initialized(false), is_prior(false), use_state_evolution_matrix(
				false), i(0), no_gpus(0), main_gpu(0), m(0), max_p(0), nsim(0), nsim_batch(0) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::HostWrapperImpl()" << ENDL;

	// check number of GPUs
	int no_devices;

	bool getDeviceCountSuccess = cudaErrchk(cudaGetDeviceCount(&no_devices));
	if (!getDeviceCountSuccess || no_devices < 1) {
		no_devices = 0;
		ERROR_LOGGER << "No cuda-enabled devices available." << ENDL;
	} else if (no_devices > MAX_NO_GPUS) {
		no_devices = MAX_NO_GPUS;
	}

	if (no_gpus <= no_devices) {
		this->no_gpus = no_gpus;
	} else {
		this->no_gpus = no_devices;
	}

	INFO_LOGGER << "Using " << this->no_gpus << " GPUs" << ENDL;

	// start devices and streams
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index, true);
		cudaErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));

		SYSDEBUG_LOGGER << "started device " << gpu_index << ENDL;

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		cudaErrchk(cudaStreamCreate(&P.stream));

		SYSDEBUG_LOGGER << "created stream on device " << gpu_index << ENDL;

		startCublas(&P.CUBLAS);

		SYSDEBUG_LOGGER << "started cublas on device " << gpu_index << ENDL;

		cublasErrchk(cublasSetStream(P.CUBLAS, P.stream));

		startCurand(&P.CURAND);

		SYSDEBUG_LOGGER << "started curand on device " << gpu_index << ENDL;

		curandErrchk(curandSetStream(P.CURAND, P.stream));

		curandErrchk(curandSetPseudoRandomGeneratorSeed(P.CURAND, 1234ULL + gpu_index * 100)); // set cuRand seed

		SYSDEBUG_LOGGER << "set curand seed on device " << gpu_index << ENDL;

		P.MEM = memory_manager_GPU(gpu_index, P.stream);
		P.MEM_evo = memory_manager_GPU(gpu_index, P.stream);
		P.MEM_sim = memory_manager_GPU(gpu_index, P.stream);

		SYSDEBUG_LOGGER << "initialized the memory managers" << ENDL;
	}
}

template<typename DOUBLE> SGDLM::HostWrapperImpl<DOUBLE>::~HostWrapperImpl() {
	SYSDEBUG_LOGGER << "HostWrapperImpl::~HostWrapperImpl()" << ENDL;

	this->clearMemory();

	// shut down GPUs
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);

		endCublas(this->simP[gpu_index].CUBLAS);

		SYSDEBUG_LOGGER << "ended cublas on device " << gpu_index << ENDL;

		endCurand(this->simP[gpu_index].CURAND);

		SYSDEBUG_LOGGER << "ended curand on device " << gpu_index << ENDL;

		cudaErrchk(cudaStreamDestroy(this->simP[gpu_index].stream));

		SYSDEBUG_LOGGER << "destroyed stream on device " << gpu_index << ENDL;

		cudaErrchk(cudaDeviceSynchronize());

		cudaErrchk(cudaDeviceReset());

		SYSDEBUG_LOGGER << "reset device " << gpu_index << ENDL;
	}
}

template<typename DOUBLE> std::size_t SGDLM::HostWrapperImpl<DOUBLE>::getNoSeries() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getNoSeries()" << ENDL;

	return this->m;
}

template<typename DOUBLE> std::size_t SGDLM::HostWrapperImpl<DOUBLE>::getMaxP() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getMaxP()" << ENDL;

	return this->max_p;
}

template<typename DOUBLE> bool SGDLM::HostWrapperImpl<DOUBLE>::getEvolutionMatrixConfiguration() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getEvolutionMatrixConfiguration()" << ENDL;

	return this->use_state_evolution_matrix;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::initMemory(std::size_t m, std::size_t max_p) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::initMemory(" << m << ", " << max_p << ")" << ENDL;

	this->clearMemory(); // call clearMemory to set simulation memory un-initialized just in case dimensions change

	this->m = m;
	this->max_p = max_p;

	if (this->no_gpus < 1) {
		return;
	}

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];
		memory_manager_GPU& MEM = P.MEM;

		// define 0,+1,-1 on device (MEM)
		P.zero = MEM.host_device_alloc<DOUBLE>(sizeof(DOUBLE));
		assignScalar<<<1, 1, 0, P.stream>>>(1, P.zero, (DOUBLE) 0);
		P.plus_one = MEM.host_device_alloc<DOUBLE>(sizeof(DOUBLE));
		assignScalar<<<1, 1, 0, P.stream>>>(1, P.plus_one, (DOUBLE) 1);
		P.minus_one = MEM.host_device_alloc<DOUBLE>(sizeof(DOUBLE));
		assignScalar<<<1, 1, 0, P.stream>>>(1, P.minus_one, (DOUBLE) -1);

		// allocate device memory for discount factors (MEM)
		P.beta = MEM.device_alloc_vec<DOUBLE>(this->m);
		P.data_delta = MEM.device_alloc_vec<DOUBLE>(this->m * this->max_p * this->max_p);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>((const DOUBLE*) P.data_delta, this->m, this->max_p * this->max_p, P.delta); // generate CPU+GPU pointer to individual matrices

		// allocate device memory for simultaneous parental sets (MEM)
		P.p = MEM.device_alloc_vec<unsigned int>(this->m);
		P.sp_indices = MEM.device_alloc_vec<unsigned int>(this->m * this->max_p);

		// allocate device memory for cache variables (MEM)
		P.Q_t = MEM.device_alloc_vec<DOUBLE>(this->m);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>(P.Q_t, this->m, 1, P.Q_t_ptrptr);
		P.e_t = MEM.device_alloc_vec<DOUBLE>(this->m);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>(P.e_t, this->m, 1, P.e_t_ptrptr);
		P.data_A_t = MEM.device_alloc_vec<DOUBLE>(this->m * this->max_p);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>(P.data_A_t, this->m, this->max_p, P.A_t); // generate CPU+GPU pointer to individual matrices

		// allocate device memory for predictors, data
		P.y_t = MEM.device_alloc_vec<DOUBLE>(this->m);
		P.data_F_t = MEM.device_alloc_vec<DOUBLE>(this->m * this->max_p);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>(P.data_F_t, this->m, this->max_p, P.F_t);

		// allocate device memory for DLM parameters (MEM)
		P.data_m_t = MEM.device_alloc_vec<DOUBLE>(this->m * this->max_p);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>((const DOUBLE*) P.data_m_t, this->m, this->max_p, P.m_t); // generate CPU+GPU pointer to individual matrices
		P.data_C_t = MEM.device_alloc_vec<DOUBLE>(this->m * this->max_p * this->max_p);
		MEM.cpyToDeviceAsPtrArray<DOUBLE>((const DOUBLE*) P.data_C_t, this->m, this->max_p * this->max_p, P.C_t); // generate CPU+GPU pointer to individual matrices
		P.n_t = MEM.device_alloc_vec<DOUBLE>(this->m);
		P.s_t = MEM.device_alloc_vec<DOUBLE>(this->m);
	}

	this->manageEvoMemory(this->use_state_evolution_matrix);

	this->memory_initialized = true;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::manageEvoMemory(bool use_state_evolution_matrix) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::initEvoMemory(" << use_state_evolution_matrix << ")" << ENDL;

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];
		memory_manager_GPU& MEM_evo = P.MEM_evo;
		memory_manager_GPU& MEM_sim = P.MEM_sim;

		// allocate device memory for evolution matrices (MEM)
		if (use_state_evolution_matrix) {
			if (!this->evo_memory_initialized) {
				P.data_G_t = MEM_evo.device_alloc_vec<DOUBLE>(this->m * this->max_p * this->max_p);
				P.G_t = NULL;
				MEM_evo.cpyToDeviceAsPtrArray<DOUBLE>(P.data_G_t, this->m, this->max_p * this->max_p, P.G_t);

				P.data_m_t_buffer = MEM_evo.device_alloc_vec<DOUBLE>(this->m * this->max_p);
				P.m_t_buffer = NULL;
				MEM_evo.cpyToDeviceAsPtrArray<DOUBLE>(P.data_m_t_buffer, this->m, this->max_p, P.m_t_buffer);

				// C_t_buffer is later initialized onto the simulation memory
			}
		} else {
			MEM_evo.clear();
			if (!this->sim_memory_initialized) { // free C_t_buffer, which is managed by MEM_sim
				MEM_sim.clear();
			}
		}
	}

	// manage outsourced memory
	if (!this->sim_memory_initialized) { // if simulation memory is not needed, clear it to free C_t_buffer; if in use, do not alter
		if (use_state_evolution_matrix) {
			this->allocate_C_t_memory();
		}
	}

	this->use_state_evolution_matrix = use_state_evolution_matrix;
	this->evo_memory_initialized = use_state_evolution_matrix;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::clearMemory() {
	SYSDEBUG_LOGGER << "HostWrapperImpl::clearGPUs()" << ENDL;

	this->memory_initialized = false;
	this->sim_memory_initialized = false;
	this->evo_memory_initialized = false;

	this->nsim = 0;
	this->nsim_batch = 0;

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);

		this->simP[gpu_index].MEM_sim.clear();

		this->simP[gpu_index].MEM_evo.clear();

		this->simP[gpu_index].MEM.clear();

		SYSDEBUG_LOGGER << "cleared memory on device " << gpu_index << ENDL;
	}
}

template<typename DOUBLE> std::size_t SGDLM::HostWrapperImpl<DOUBLE>::getNSim() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getNSim()" << ENDL;

	return this->nsim;
}

template<typename DOUBLE> std::size_t SGDLM::HostWrapperImpl<DOUBLE>::getNSimBatch() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getNSimBatch()" << ENDL;

	return this->nsim_batch;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::initSimMemory(std::size_t nsim, std::size_t nsim_batch) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::initSimMemory(" << nsim << ", " << nsim_batch << ")" << ENDL;
	myAssert(this->checkInitialized());

	// set new dimensions
	nsim /= this->no_gpus;
	this->nsim = nsim * this->no_gpus;
	this->nsim_batch = nsim_batch > this->nsim ? this->nsim : nsim_batch;

	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		memory_manager_GPU& MEM_sim = P.MEM_sim;

		MEM_sim.clear(); // free existing memory before allocating new memory

		P.nsim = nsim;

		// FOR VB POSTERIOR ESTIMATION AND FORECASTING
		/*
		 * lambdas is organized by batch     : n-array of pointers to m-arrays
		 * randoms is organized by batch     : n-array of pointers to (m * max_p)-arrays
		 * thetas is organized by batch      : n-array of pointers to (m * max_p)-arrays
		 * Gammas is organized by batch      : Gammas_batch_size-array of pointers to (m * m)-arrays
		 * chol_C_t is organized by dimension: m-array of (max_p * max_p)-arrays
		 * LU_infos is organized by batch    : Gammas_batch_size-array of scalars
		 * LU_pivots is organized by batch   : Gammas_batch_size-array of m-arrays
		 *
		 *
		 * chol_C_t_nrepeat_ptr = n x [chol(C_t[0]), chol(C_t[1]), ..., chol(C_t[m-1])]
		 * randoms_nrepeat_ptr  = same logic as thetas_nrepeat_ptr---just pointing to randoms instead
		 * thetas_nrepeat_ptr   = thetas[0], thetas[max_p], thetas[2*max_p], ..., thetas[(m-1)*max_p], ..., thetas[n*(m-1)*max_p]
		 */

		// allocate device memory
		P.data_lambdas = MEM_sim.device_alloc_vec<DOUBLE>(this->m * P.nsim);
		P.data_randoms = MEM_sim.device_alloc_vec<DOUBLE>(((this->max_p > 4) ? this->max_p : 4) * this->m * P.nsim); // allocate max(4, least max_p)*m*P.nsim so that memory P.data_random_pt2 still has 2*m*P.nsim entries allocated
		P.data_randoms_pt2 = (DOUBLE*) ((char*) P.data_randoms + (2 * this->m * P.nsim) * sizeof(DOUBLE)); //&data_randoms[2 * m * nsim]; //TODO: verify
		P.data_thetas = MEM_sim.device_alloc_vec<DOUBLE>(this->max_p * this->m * P.nsim);
		P.data_Gammas = MEM_sim.device_alloc_vec<DOUBLE>(this->m * this->m * 2 * nsim_batch); // allocate for 2 * nsim_batch: VB_posterior can use double the batch size and forecasting will use the 2nd half for the inverse
		P.data_chol_C_t = MEM_sim.device_alloc_vec<DOUBLE>(this->max_p * this->max_p * this->m);
		P.LU_pivots = MEM_sim.device_alloc_vec<int>(this->m * 2 * nsim_batch);
		P.LU_infos = MEM_sim.device_alloc_vec<int>(2 * nsim_batch);

		// define array pointers
		P.lambdas = NULL;
		P.randoms_nrepeat_ptr = NULL;
		P.Gammas = NULL;
		P.thetas = NULL;
		P.thetas_nrepeat_ptr = NULL;
		P.lambdas_nrepeat_ptr = NULL;
		P.chol_C_t = NULL;
		P.chol_C_t_nrepeat_ptr = NULL;

		// assign repeat pointers
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_lambdas, P.nsim, this->m, P.lambdas);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_Gammas, 2 * nsim_batch, m * m, P.Gammas);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_chol_C_t, this->m, this->max_p * this->max_p, P.chol_C_t);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_thetas, P.nsim, this->m * this->max_p, P.thetas);

		// assign nrepeat pointers
		MEM_sim.cpyToDeviceAsPtrArrayByCol<DOUBLE>(P.data_randoms, P.nsim, this->m, this->max_p, P.randoms_nrepeat_ptr);
		MEM_sim.cpyToDeviceAsPtrArrayRepeatByBatch<DOUBLE>(P.data_chol_C_t, this->m, this->max_p * this->max_p, P.nsim,
				P.chol_C_t_nrepeat_ptr);
		MEM_sim.cpyToDeviceAsPtrArrayByCol<DOUBLE>(P.data_thetas, P.nsim, this->m, this->max_p, P.thetas_nrepeat_ptr);
		MEM_sim.cpyToDeviceAsPtrArrayByCol<DOUBLE>(P.data_lambdas, P.nsim, this->m, 1, P.lambdas_nrepeat_ptr);

		// FOR VB POSTERIOR ESTIMATION
		/*
		 * IS_weights is organized by batch          : n-array of scalars
		 * mean_lambdas is organized by dimension    : m-array of scalars
		 * mean_log_lambdas is organized by dimension: m-array of scalars
		 * mean_n_t is organized by dimension        : m-array of scalars
		 * mean_s_t is organized by dimension        : m-array of scalars
		 * mean_Q_t is organized by dimension        : m-array of scalars
		 * mean_m_t is organized by dimension        : m-array of max_p-arrays
		 * mean_C_t is organized by dimension        : m-array of (max_p * max_p)-arrays
		 * C_t_buffer is organized by dimension      : m-array of (max_p * max_p)-arrays
		 * INV_infos is organized by batch           : m-array of scalars
		 * INV_pivots is organized by batch          : m-array of max_p-arrays
		 *
		 *
		 * chol_C_t_nrepeat_ptr = n x [chol(C_t[0]), chol(C_t[1]), ..., chol(C_t[m-1])]
		 * randoms_nrepeat_ptr  = same logic as thetas_nrepeat_ptr---just pointing to randoms instead
		 * thetas_nrepeat_ptr   = thetas[0], thetas[max_p], thetas[2*max_p], ..., thetas[(m-1)*max_p], ..., thetas[n*(m-1)*max_p]
		 */

		// allocate device memory
		P.IS_weights = MEM_sim.device_alloc_vec<DOUBLE>(P.nsim);
		P.sum_det_weights = MEM_sim.device_alloc_vec<DOUBLE>(1);
		P.INV_pivots = MEM_sim.device_alloc_vec<int>(this->max_p * this->m);
		P.INV_infos = MEM_sim.device_alloc_vec<int>(this->m);
		P.mean_lambdas = MEM_sim.device_alloc_vec<DOUBLE>(this->m);
		P.mean_log_lambdas = MEM_sim.device_alloc_vec<DOUBLE>(this->m);
		P.data_mean_m_t = MEM_sim.device_alloc_vec<DOUBLE>(this->max_p * this->m);
		P.data_mean_C_t = MEM_sim.device_alloc_vec<DOUBLE>(this->max_p * this->max_p * this->m);
		P.mean_n_t = MEM_sim.device_alloc_vec<DOUBLE>(this->m);
		P.mean_s_t = MEM_sim.device_alloc_vec<DOUBLE>(this->m);
		P.mean_Q_t = MEM_sim.device_alloc_vec<DOUBLE>(this->m);
		P.data_C_t_buffer = MEM_sim.device_alloc_vec<DOUBLE>(this->m * this->max_p * this->max_p);

		// define array pointers
		P.mean_m_t = NULL;
		P.mean_C_t = NULL;
		P.C_t_buffer = NULL;

		// assign array pointers
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_mean_m_t, this->m, this->max_p, P.mean_m_t);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_mean_C_t, this->m, this->max_p * this->max_p, P.mean_C_t);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_C_t_buffer, this->m, this->max_p * this->max_p, P.C_t_buffer);

		// FOR FORECASTING
		/*
		 * y is organized by batch           : n-array of pointers to m-arrays
		 * nus is organized by batch         : n-array of pointers to m-arrays
		 * Gammas_inv is organized by batch  : Gammas_batch_size-array of pointers to (m * m)-arrays
		 */

		// allocate device memory
		P.data_x_t = MEM_sim.device_alloc_vec<DOUBLE>(this->m * this->max_p);
		P.data_y = MEM_sim.device_alloc_vec<DOUBLE>(this->m * P.nsim);
		P.data_nus = MEM_sim.device_alloc_vec<DOUBLE>(this->m * P.nsim);
		P.data_Gammas_inv = (DOUBLE*) ((char*) P.data_Gammas + (this->m * this->m * nsim_batch) * sizeof(DOUBLE)); //MEM_sim.device_alloc<DOUBLE>(3, dim_Gammas);

		// define array pointers
		P.x_t = NULL;
		P.y = NULL;
		P.nus = NULL;
		P.Gammas_inv = NULL;

		// assign array pointers
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_x_t, this->m, this->max_p, P.x_t);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_y, P.nsim, this->m, P.y);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_nus, P.nsim, this->m, P.nus);
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_Gammas_inv, nsim_batch, this->m * this->m, P.Gammas_inv);
	}

	this->sim_memory_initialized = true;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::clearSimMemory() {
	SYSDEBUG_LOGGER << "HostWrapperImpl::clearSimMemory()" << ENDL;
	myAssert(this->checkInitialized());

	this->sim_memory_initialized = false;

	this->nsim = 0;
	this->nsim_batch = 0;

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);

		this->simP[gpu_index].MEM_sim.clear();

		SYSDEBUG_LOGGER << "cleared simulation memory on device " << gpu_index << ENDL;
	}

	if (this->evo_memory_initialized) { // re-allocate C_t_buffer
		this->allocate_C_t_memory();
	}
}

template<typename DOUBLE> bool SGDLM::HostWrapperImpl<DOUBLE>::isPrior() const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::isPrior()" << ENDL;

	return this->is_prior;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::isPrior(bool is_prior) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::isPrior(" << is_prior << ")" << ENDL;

	this->is_prior = is_prior;
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::getParameters(DOUBLE* host_data_m_t,
		DOUBLE* host_data_C_t, DOUBLE* host_data_n_t, DOUBLE* host_data_s_t) const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getParameters()" << ENDL;
	myAssert(this->checkInitialized());

	startCuda(this->main_gpu);
	const simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[this->main_gpu];

	if (host_data_m_t != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_m_t, host_data_m_t, this->m * this->max_p, P.stream);
	}

	if (host_data_C_t != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_C_t, host_data_C_t, this->m * this->max_p * this->max_p, P.stream);
	}

	if (host_data_n_t != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.n_t, host_data_n_t, this->m, P.stream);
	}

	if (host_data_s_t != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.s_t, host_data_s_t, this->m, P.stream);
	}

	cudaErrchk(cudaStreamSynchronize(this->simP[this->main_gpu].stream));
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::setParameters(const DOUBLE* host_data_m_t,
		const DOUBLE* host_data_C_t, const DOUBLE* host_data_n_t, const DOUBLE* host_data_s_t) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::setParameters()" << ENDL;
	myAssert(this->checkInitialized());

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		if (host_data_m_t != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_m_t, host_data_m_t, this->m * this->max_p, P.stream);
		}

		if (host_data_C_t != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_C_t, host_data_C_t, this->m * this->max_p * this->max_p,
					P.stream);
		}

		if (host_data_n_t != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.n_t, host_data_n_t, this->m, P.stream);
		}

		if (host_data_s_t != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.s_t, host_data_s_t, this->m, P.stream);
		}
	}
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::getDiscountFactors(DOUBLE* host_data_beta,
		DOUBLE* host_data_delta) const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getDiscountFactors()" << ENDL;
	myAssert(this->checkInitialized());

	startCuda(this->main_gpu);
	const simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[this->main_gpu];

	if (host_data_beta != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.beta, host_data_beta, this->m, P.stream);
	}

	if (host_data_delta != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_delta, host_data_delta, this->m * this->max_p * this->max_p,
				P.stream);
	}

	cudaErrchk(cudaStreamSynchronize(this->simP[this->main_gpu].stream));
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::setDiscountFactors(const DOUBLE* host_data_beta,
		const DOUBLE* host_data_delta) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::setDiscountFactors()" << ENDL;
	myAssert(this->checkInitialized());

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		if (host_data_beta != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.beta, host_data_beta, this->m, P.stream);
		}

		if (host_data_delta != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_delta, host_data_delta, this->m * this->max_p * this->max_p,
					P.stream);
		}
	}
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::getEvolutionMatrix(DOUBLE* host_data_G_t) const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getEvolutionMatrix()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkUseStateEvolutionMatrix());

	startCuda(this->main_gpu);
	const simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[this->main_gpu];

	if (host_data_G_t != NULL) {
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_G_t, host_data_G_t, this->m * this->max_p * this->max_p, P.stream);
	}

	cudaErrchk(cudaStreamSynchronize(this->simP[this->main_gpu].stream));
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::setEvolutionMatrix(const DOUBLE* host_data_G_t) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::setEvolutionMatrix()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkUseStateEvolutionMatrix());

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		if (host_data_G_t != NULL) {
			memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_G_t, host_data_G_t, this->m * this->max_p * this->max_p,
					P.stream);
		}
	}
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::getParentalSets(unsigned int* host_data_p,
		unsigned int* host_data_sp_indices) const {
	SYSDEBUG_LOGGER << "HostWrapperImpl::getParentalSets()" << ENDL;
	myAssert(this->checkInitialized());

	startCuda(this->main_gpu);
	const simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[this->main_gpu];

	if (host_data_p != NULL) {
		memory_manager_GPU::cpyToHost<unsigned int>(P.p, host_data_p, this->m, P.stream);
	}

	if (host_data_sp_indices != NULL) {
		memory_manager_GPU::cpyToHost<unsigned int>(P.sp_indices, host_data_sp_indices, this->m * this->max_p,
				P.stream);
	}

	cudaErrchk(cudaStreamSynchronize(this->simP[this->main_gpu].stream));
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::setParentalSets(const unsigned int* host_data_p,
		const unsigned int* host_data_sp_indices) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::setParentalSets()" << ENDL;
	myAssert(this->checkInitialized());

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		if (host_data_p != NULL) {
			memory_manager_GPU::cpyToDevice<unsigned int>(P.p, host_data_p, this->m, P.stream);
		}

		if (host_data_sp_indices != NULL) {
			memory_manager_GPU::cpyToDevice<unsigned int>(P.sp_indices, host_data_sp_indices, this->m * this->max_p,
					P.stream);
		}
	}
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::computePrior() {
	SYSDEBUG_LOGGER << "HostWrapperImpl::computePrior()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkPrior(false));

	// call SGDLM::compute_one_step_ahead_prior with G_t = NULL
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		SGDLM<DOUBLE>::compute_one_step_ahead_prior(this->m, this->max_p, P.p, P.m_t, P.C_t, P.n_t, P.s_t, P.beta,
				(const DOUBLE**) P.delta, P.stream);
	}

	// wait for results
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		cudaErrchk(cudaStreamSynchronize(this->simP[gpu_index].stream));
	}

	this->isPrior(true);
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::computePrior(const DOUBLE* host_data_G_t) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::computePrior(...)" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkPrior(false));
	myAssert(this->checkUseStateEvolutionMatrix());

	this->setEvolutionMatrix(host_data_G_t);

	// call SGDLM::compute_one_step_ahead_prior with the current state evolution matrix
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		SGDLM<DOUBLE>::compute_one_step_ahead_prior(this->m, this->max_p, P.p, P.m_t, P.C_t, P.n_t, P.s_t, P.beta,
				(const DOUBLE**) P.delta, P.stream, P.CUBLAS, P.zero, P.plus_one, (const DOUBLE**) P.G_t, P.C_t_buffer,
				P.m_t_buffer);
	}

	// wait for results
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		cudaErrchk(cudaStreamSynchronize(this->simP[gpu_index].stream));
	}

	this->isPrior(true);
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::computeForecast(DOUBLE* host_data_ytp1,
		const DOUBLE* host_data_x_tp1) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::computeForecast()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkSimInitialized());
	myAssert(this->checkPrior(true));

	// copy predictors onto device memory
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_x_t, host_data_x_tp1, this->m * this->max_p, P.stream);
	}

	// compute forecasts
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		SGDLM<DOUBLE>::forecast((const DOUBLE*) P.zero, (const DOUBLE*) P.plus_one, this->m, this->max_p,
				(const unsigned int*) P.p, (const unsigned int*) P.sp_indices, (const DOUBLE**) P.m_t,
				(const DOUBLE**) P.C_t, (const DOUBLE*) P.n_t, (const DOUBLE*) P.s_t, P.nsim, this->nsim_batch,
				(const DOUBLE**) P.x_t, P.y, P.data_nus, P.nus, P.lambdas, P.data_randoms, P.data_randoms_pt2,
				P.randoms_nrepeat_ptr, P.Gammas, P.Gammas_inv, P.LU_pivots, P.LU_infos, P.chol_C_t,
				P.chol_C_t_nrepeat_ptr, P.thetas, P.thetas_nrepeat_ptr, P.stream, P.CUBLAS, P.CURAND);
	}

	// copy results on host memory
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		// copy the result into the right place of the output array
		DOUBLE* out_ptr_pos = (DOUBLE*) ((char*) host_data_ytp1 + gpu_index * this->m * P.nsim * sizeof(DOUBLE));
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_y, out_ptr_pos, this->m * P.nsim, P.stream);
	}

	// wait until computations and memory transfers are complete
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		cudaErrchk(cudaStreamSynchronize(this->simP[gpu_index].stream));
	}
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::computePosterior(const DOUBLE* host_data_y_t,
		const DOUBLE* host_data_F_t) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::computePosterior()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkPrior(true));

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		memory_manager_GPU::cpyToDevice<DOUBLE>(P.y_t, host_data_y_t, this->m, P.stream);
		memory_manager_GPU::cpyToDevice<DOUBLE>(P.data_F_t, host_data_F_t, this->m * this->max_p, P.stream);
	}

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		SGDLM<DOUBLE>::compute_posterior(P.zero, P.plus_one, P.minus_one, this->m, this->max_p, P.p, P.m_t, P.C_t,
				P.n_t, P.s_t, (const DOUBLE*) P.y_t, (const DOUBLE**) P.F_t, P.Q_t, P.e_t, P.A_t, P.Q_t_ptrptr,
				P.e_t_ptrptr, P.CUBLAS, P.stream);
	}

	// wait for results
	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		cudaErrchk(cudaStreamSynchronize(this->simP[gpu_index].stream));
	}

	this->isPrior(false);
}

template<typename DOUBLE> void SGDLM::HostWrapperImpl<DOUBLE>::computeVBPosterior(DOUBLE* host_data_mean_m_t,
		DOUBLE* host_data_mean_C_t, DOUBLE* host_data_mean_n_t, DOUBLE* host_data_mean_s_t,
		DOUBLE* host_data_IS_weights, DOUBLE* host_sum_det_weights) {
	SYSDEBUG_LOGGER << "HostWrapperImpl::runVB()" << ENDL;
	myAssert(this->checkInitialized());
	myAssert(this->checkSimInitialized());
	myAssert(this->checkPrior(false));

	size_t batch_multiplier = 2;

	SYSDEBUG_LOGGER << "before starting VB simulation on all selected GPUs" << ENDL;

	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		SGDLM<DOUBLE>::VB_posterior((const DOUBLE*) P.zero, (const DOUBLE*) P.plus_one, this->m, this->max_p,
				(const unsigned int*) P.p, (const unsigned int*) P.sp_indices, (const DOUBLE**) P.m_t,
				(const DOUBLE**) P.C_t, (const DOUBLE*) P.n_t, (const DOUBLE*) P.s_t, P.nsim, P.lambdas, P.data_randoms,
				P.data_randoms_pt2, P.randoms_nrepeat_ptr, P.Gammas, batch_multiplier * this->nsim_batch, P.IS_weights,
				P.sum_det_weights, P.chol_C_t, P.chol_C_t_nrepeat_ptr, P.thetas, P.thetas_nrepeat_ptr, P.LU_pivots,
				P.LU_infos, P.mean_lambdas, P.mean_log_lambdas, P.mean_m_t, P.mean_C_t, P.C_t_buffer, P.INV_pivots,
				P.INV_infos, P.mean_n_t, P.mean_s_t, P.mean_Q_t, P.lambdas, P.lambdas_nrepeat_ptr, P.stream, P.CUBLAS,
				P.CURAND);
	}

	SYSDEBUG_LOGGER << "VB simulation initiated" << ENDL;

	// allocate temporary host memory to copy data in from every gpu
	memory_manager host_MEM_temp;
	DOUBLE* host_temp_mean_m_t = host_MEM_temp.host_alloc_vec<DOUBLE>(this->no_gpus * this->m * this->max_p);
	DOUBLE* host_temp_mean_C_t = host_MEM_temp.host_alloc_vec<DOUBLE>(this->no_gpus * this->m * this->max_p * this->max_p);
	DOUBLE* host_temp_mean_n_t = host_MEM_temp.host_alloc_vec<DOUBLE>(this->no_gpus * this->m);
	DOUBLE* host_temp_mean_s_t = host_MEM_temp.host_alloc_vec<DOUBLE>(this->no_gpus * this->m);
	DOUBLE* host_temp_IS_weights = host_MEM_temp.host_alloc_vec<DOUBLE>(this->nsim);
	DOUBLE* host_temp_sum_det_weights = host_MEM_temp.host_alloc_vec<DOUBLE>(this->no_gpus);

	// retrieve VB results into temporary host memory
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);

		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];

		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_mean_m_t, &host_temp_mean_m_t[gpu_index * this->m * this->max_p],
				this->m * this->max_p, P.stream);
		memory_manager_GPU::cpyToHost<DOUBLE>(P.data_mean_C_t,
				&host_temp_mean_C_t[gpu_index * this->m * this->max_p * this->max_p],
				this->m * this->max_p * this->max_p, P.stream);
		memory_manager_GPU::cpyToHost<DOUBLE>(P.mean_n_t, &host_temp_mean_n_t[gpu_index * this->m], this->m,
				P.stream);
		memory_manager_GPU::cpyToHost<DOUBLE>(P.mean_s_t, &host_temp_mean_s_t[gpu_index * this->m], this->m,
				P.stream);
		memory_manager_GPU::cpyToHost<DOUBLE>(P.IS_weights, &host_temp_IS_weights[gpu_index * P.nsim], P.nsim,
				P.stream);
		memory_manager_GPU::cpyToHost<DOUBLE>(P.sum_det_weights, &host_temp_sum_det_weights[gpu_index], 1, P.stream);
	}

	SYSDEBUG_LOGGER << "waiting for memory transfer to complete" << ENDL;

	// wait until all results are downloaded
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		startCuda(gpu_index);
		cudaErrchk(cudaStreamSynchronize(this->simP[gpu_index].stream));

		SYSDEBUG_LOGGER << "synchronized stream on device " << gpu_index << ENDL;
	}

	// sum up determinant weights
	host_sum_det_weights[0] = 0;
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "the sum of the determinants of GPU " << gpu_index << " is: "
				<< host_temp_sum_det_weights[gpu_index] << ENDL;

		host_sum_det_weights[0] += host_temp_sum_det_weights[gpu_index];
	}

	SYSDEBUG_LOGGER << "host_sum_det_weights[0] = " << host_sum_det_weights[0] << ENDL;

	// average the means from the different GPUs
	for (size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		SYSDEBUG_LOGGER << "gpu_index = " << gpu_index << ENDL;

		// calculate weight of the batch produced by this GPU
		DOUBLE weight_scale = host_temp_sum_det_weights[gpu_index] / host_sum_det_weights[0];

		SYSDEBUG_LOGGER << "weight_scale = " << host_temp_sum_det_weights[gpu_index] << " / " << host_sum_det_weights[0]
				<< " = " << weight_scale << ENDL;

		// write mean_m_t
		for (size_t i = 0; i < this->m * this->max_p; i++) {
			if (gpu_index == 0) { // initialize memory to 0
				host_data_mean_m_t[i] = 0;
			}
			host_data_mean_m_t[i] += weight_scale * host_temp_mean_m_t[gpu_index * this->m * this->max_p + i];
		}

		// write mean_C_t
		for (size_t i = 0; i < this->m * this->max_p * this->max_p; i++) {
			if (gpu_index == 0) { // initialize memory to 0
				host_data_mean_C_t[i] = 0;
			}

			host_data_mean_C_t[i] += weight_scale * host_temp_mean_C_t[gpu_index * this->m * this->max_p * this->max_p + i];
		}

		// write mean_n_t, mean_s_t
		for (size_t i = 0; i < this->m; i++) {
			if (gpu_index == 0) { // initialize memory to 0
				host_data_mean_n_t[i] = 0;
				host_data_mean_s_t[i] = 0;
			}

			host_data_mean_n_t[i] += weight_scale * host_temp_mean_n_t[gpu_index * this->m + i];
			host_data_mean_s_t[i] += weight_scale * host_temp_mean_s_t[gpu_index * this->m + i];
		}

		// write IS_weights
		DOUBLE sum_weights = 0;
		for (size_t i = 0; i < this->simP[gpu_index].nsim; i++) {
			host_data_IS_weights[gpu_index * this->simP[gpu_index].nsim + i] = weight_scale
					* host_temp_IS_weights[gpu_index * this->simP[gpu_index].nsim + i];
			sum_weights += host_data_IS_weights[gpu_index * this->simP[gpu_index].nsim + i];
		}

		SYSDEBUG_LOGGER << "sum_weights = " << sum_weights << ENDL;
	}

	SYSDEBUG_LOGGER << "before clearing host_MEM_temp" << ENDL;

	host_MEM_temp.clear();
}

/*
 *
 *
 *
 *
 *
 *
 */

template<typename DOUBLE> inline bool SGDLM::HostWrapperImpl<DOUBLE>::checkInitialized() const {
	if (!this->memory_initialized) {
		ERROR_LOGGER << "The device memory is not initialized." << ENDL;
		return false;
	}
	return true;
}

template<typename DOUBLE> inline bool SGDLM::HostWrapperImpl<DOUBLE>::checkSimInitialized() const {
	if (!this->sim_memory_initialized) {
		ERROR_LOGGER << "The simulation device memory is not initialized." << ENDL;
		return false;
	}
	return true;
}

template<typename DOUBLE> inline bool SGDLM::HostWrapperImpl<DOUBLE>::checkUseStateEvolutionMatrix() const {
	if (!this->use_state_evolution_matrix) {
		ERROR_LOGGER << "The use of state evolution matrices is disabled." << ENDL;
		return false;
	}
	return true;
}

template<typename DOUBLE> inline bool SGDLM::HostWrapperImpl<DOUBLE>::checkPrior(bool is_prior) const {
	if (is_prior) {
		if (!this->is_prior) {
			ERROR_LOGGER << "This function can only be executed when the parameters are prior parameters." << ENDL;
			return false;
		}
		return true;
	} else {
		if (this->is_prior) {
			ERROR_LOGGER << "This function can only be executed when the parameters are posterior parameters." << ENDL;
			return false;
		}
		return true;
	}
}

/*
 *
 *
 *
 *
 *
 *
 */

template<typename DOUBLE> inline void SGDLM::HostWrapperImpl<DOUBLE>::allocate_C_t_memory() { // allocate C_t_buffer to simulation memory
	SYSDEBUG_LOGGER << "HostWrapperImpl::allocate_C_t_memory()" << ENDL;

	for (std::size_t gpu_index = 0; gpu_index < this->no_gpus; gpu_index++) {
		startCuda(gpu_index);
		simPointers<DOUBLE, memory_manager_GPU>& P = this->simP[gpu_index];
		memory_manager_GPU& MEM_sim = P.MEM_sim;

		P.data_C_t_buffer = MEM_sim.device_alloc_vec<DOUBLE>(this->m * this->max_p * this->max_p);
		P.C_t_buffer = NULL;
		MEM_sim.cpyToDeviceAsPtrArray<DOUBLE>(P.data_C_t_buffer, this->m, this->max_p * this->max_p, P.C_t_buffer);
	}
}

/*
 *
 *
 *
 *
 *
 *
 */

// explicit instantiation
template class SGDLM::HostWrapperImpl<DOUBLETYPE>;
