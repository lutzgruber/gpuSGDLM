#include <curand.h>
#include <cublas_v2.h>

#include "SGDLM/SGDLM.cuh"
#include "loggerGPU.cuh"
#include "cublas_manager.cuh"
#include "curand_manager.cuh"
#include "kernel_functions.cuh"


template<typename DOUBLE> void SGDLM::SGDLM<DOUBLE>::compute_posterior(const DOUBLE* zero, const DOUBLE* plus_one,
		const DOUBLE* minus_one, size_t m, size_t max_p, const unsigned int* p, DOUBLE** m_t, DOUBLE** C_t, DOUBLE* n_t,
		DOUBLE* s_t, const DOUBLE* const y_t, const DOUBLE** const F_t, DOUBLE* Q_t, DOUBLE* e_t, DOUBLE** A_t,
		DOUBLE** Q_t_ptrptr, DOUBLE** e_t_ptrptr, cublasHandle_t CUBLAS, cudaStream_t stream) {
	SYSDEBUG_LOGGER << "SGDLM::compute_posterior()" << ENDL;

	cublasErrchk(cublasSetStream(CUBLAS, stream));

	// cut off m_t at p
	dim3 threadsPerBlock(THREADS_PER_BLOCK / max_p, max_p);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
	batchedCutOffVector<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, p, m_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after cutting off m_t after p" << ENDL;

	// cut off C_t at p x p
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
	batchedCutOffMatrix<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, max_p, p, p, C_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after cutting off C_t outside of the p x p matrix" << ENDL;

	// A_t = R_t * F_t
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, max_p, 1, max_p, plus_one, (const DOUBLE**) C_t, max_p, F_t,
			max_p, zero, A_t, max_p, m);
	SYSDEBUG_LOGGER << "... after A_t = R_t * F_t" << ENDL;

	// Q_t = c_t + F_t' * A_t
	copy<<<(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m, (const DOUBLE*) s_t,
			Q_t); // copy values from s_t into Q_t; used to be: cublasXcopy(CUBLAS, m, (const DOUBLE*) s_t, 1, Q_t, 1);
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, max_p, plus_one, (const DOUBLE**) F_t, max_p,
			(const DOUBLE**) A_t, max_p, plus_one, Q_t_ptrptr, 1, m);
	SYSDEBUG_LOGGER << "... after Q_t = c_t + F_t' * A_t" << ENDL;

	// A_t = A_t / Q_t
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
	//SYSDEBUG_LOGGER << "... threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << ENDL;
	//SYSDEBUG_LOGGER << "... numBlocks = (" << numBlocks.x << ", " << numBlocks.y << ")" << ENDL;
	batchedScale<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, A_t, (const DOUBLE*) Q_t,
			SCALE_TRANSFORMATION_INV);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after A_t = A_t / Q_t" << ENDL;

	// e_t = y_t - F_t' * a_t
	copy<<<(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m, y_t, e_t); // copy values from y_t into e_t; used to be: cublasXcopy(CUBLAS, m, y_t, 1, e_t, 1);
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, max_p, minus_one, F_t, max_p, (const DOUBLE**) m_t,
			max_p, plus_one, e_t_ptrptr, 1, m);
	SYSDEBUG_LOGGER << "... after e_t = y_t - F_t' * a_t" << ENDL;

	// n_t = r_t + 1
	addScalar<<<(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m, n_t, (DOUBLE) 1.0);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after n_t = r_t + 1" << ENDL;

	// s_t = c_t * (r_t + e_t^2 / Q_t) / n_t
	computeSt<<<(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m, s_t,
			(const DOUBLE*) n_t, (const DOUBLE*) e_t, (const DOUBLE*) Q_t);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after s_t = c_t * (r_t + e_t^2 / Q_t) / n_t" << ENDL;

	// m_t = a_t + A_t * e_t
	batchedComputeMt<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, m_t, (const DOUBLE**) A_t,
			(const DOUBLE*) e_t);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after m_t = a_t + A_t * e_t" << ENDL;

	// C_t = R_t - Q_t * A_t * A_t'
	batchedScale<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, A_t, (const DOUBLE*) Q_t,
			SCALE_TRANSFORMATION_SQRT); // set A_t = sqrt(Q_t) * A_t
	cudaErrchk(cudaGetLastError());
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_T, max_p, max_p, 1, minus_one, (const DOUBLE**) A_t, max_p,
			(const DOUBLE**) A_t, max_p, plus_one, C_t, max_p, m);
	SYSDEBUG_LOGGER << "... after C_t = R_t - Q_t * A_t * A_t'" << ENDL;

	// C_t = C_t * s_t/c_t
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
	batchedScaleCt<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p * max_p, C_t, n_t, e_t, Q_t);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after C_t = C_t * (s_t/c_t)" << ENDL;

}

template<typename DOUBLE> void SGDLM::SGDLM<DOUBLE>::compute_one_step_ahead_prior(size_t m, size_t max_p,
		const unsigned int* p, DOUBLE** m_t, DOUBLE** C_t, DOUBLE* n_t, DOUBLE* s_t, const DOUBLE* beta,
		const DOUBLE** delta, cudaStream_t stream, cublasHandle_t CUBLAS, const DOUBLE* zero, const DOUBLE* plus_one,
		const DOUBLE** G_t, DOUBLE** C_t_buffer, DOUBLE** m_t_buffer) {
	SYSDEBUG_LOGGER << "SGDLM::compute_one_step_ahead_prior()" << ENDL;

	// r_t = beta * n_t
	multiply<<<(m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m, n_t, beta); // replaced: cublasXscal(CUBLAS, m, (const DOUBLE*) beta, n_t, 1);
	cudaErrchk(cudaGetLastError());
	SYSDEBUG_LOGGER << "... after r_t = beta * n_t" << ENDL;

	// R_t = C_t / delta
	dim3 threadsPerBlock(THREADS_PER_BLOCK / max_p, max_p);
	dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
	SYSDEBUG_LOGGER << "... threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << ENDL;
	SYSDEBUG_LOGGER << "... numBlocks = (" << numBlocks.x << ", " << numBlocks.y << ")" << ENDL;
	//batchedScale<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, p, p, C_t, delta, SCALE_TRANSFORMATION_INV);
	//batchedMatrixDiagScale<<<numBlocks, threadsPerBlock, threadsPerBlock.x * max_p * sizeof(DOUBLE), stream>>>(m, max_p, p, p, C_t, delta, SCALE_TRANSFORMATION_INV);
	batchedMatrixScale<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, p, p, C_t, delta, SCALE_TRANSFORMATION_INV); // CHANGED ON 2014/9/23
	cudaErrchk(cudaGetLastError());
	//batchedScale<<<numBlocks, threadsPerBlock>>>(m, max_p * max_p, C_t, (const DOUBLE*) delta, SCALE_TRANSFORMATION_INV);
	SYSDEBUG_LOGGER << "... after R_t = C_t / delta" << ENDL;

	if (G_t != NULL) {
		// a_t = G_t * m_t
		threadsPerBlock = dim3(max_p, THREADS_PER_BLOCK / max_p);
		numBlocks = dim3(1, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
		batchedCopy<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, (const DOUBLE**) m_t, m_t_buffer);
		cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, max_p, 1, max_p, plus_one, G_t, max_p,
				(const DOUBLE**) m_t_buffer, max_p, zero, m_t, max_p, m);
		threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
		numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
		batchedCutOffVector<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, p, m_t);
		SYSDEBUG_LOGGER << "... after a_t = G_t * m_t" << ENDL;

		// R_t = G_t * R_t * G_t'
		cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, max_p, max_p, max_p, plus_one, G_t, max_p,
				(const DOUBLE**) C_t, max_p, zero, C_t_buffer, max_p, m);
		cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_T, max_p, max_p, max_p, plus_one, (const DOUBLE**) C_t_buffer,
				max_p, G_t, max_p, zero, C_t, max_p, m);
		threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
		numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
		batchedCutOffMatrix<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, max_p, p, p, C_t);
		SYSDEBUG_LOGGER << "... after R_t = G_t * R_t * G_t'" << ENDL;
	}
}

template<typename DOUBLE> void SGDLM::SGDLM<DOUBLE>::VB_posterior(const DOUBLE* zero, const DOUBLE* plus_one, size_t m,
		size_t max_p, const unsigned int* p, const unsigned int* sp_indices, const DOUBLE** m_t, const DOUBLE** C_t,
		const DOUBLE* n_t, const DOUBLE* s_t, size_t n, DOUBLE** lambdas, DOUBLE* randoms, DOUBLE* randoms_pt2,
		DOUBLE** randoms_nrepeat_ptr, DOUBLE** Gammas, size_t Gammas_batch_size, DOUBLE* IS_weights,
		DOUBLE* sum_unnormalized_IS_weights, DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas,
		DOUBLE** thetas_nrepeat_ptr, int* LU_pivots, int* LU_infos, DOUBLE* mean_lambdas, DOUBLE* mean_log_lambdas,
		DOUBLE** mean_m_t, DOUBLE** mean_C_t, DOUBLE** C_t_buffer, int* INV_pivots, int* INV_infos, DOUBLE* mean_n_t,
		DOUBLE* mean_s_t, DOUBLE* mean_Q_t, DOUBLE** Q_t, DOUBLE** Q_t_array_ptr, cudaStream_t stream,
		cublasHandle_t CUBLAS, curandGenerator_t CURAND, bool do_forecast, const DOUBLE** const x_tp1,
		DOUBLE** y_tp1_nrepeat_ptr, DOUBLE* data_nus, DOUBLE** nus_nrepeat_ptr, DOUBLE** Gammas_inv) {
	SYSDEBUG_LOGGER << "SGDLM::VB_posterior()" << ENDL;

	cublasErrchk(cublasSetStream(CUBLAS, stream));

	// sample lambdas ~ Gamma(n_t/2, n_t*s_t/2); thetas ~ N(m_t, C_t/(s_t*lambda_t))
	SGDLM<DOUBLE>::sample_parameters(zero, plus_one, m, max_p, p, m_t, C_t, n_t, s_t, n, lambdas, randoms, randoms_pt2,
			randoms_nrepeat_ptr, chol_C_t, chol_C_t_nrepeat_ptr, thetas, thetas_nrepeat_ptr, stream, CUBLAS, CURAND);

	dim3 threadsPerBlock, numBlocks;

	if (do_forecast) {
		// sample nus ~ N(0,1)
		curandGenerateNormalX(CURAND, data_nus, n * m, 0, 1);

		SYSDEBUG_LOGGER << "after sampling nus ~ N(0,1)" << ENDL;

		// compute nu = (nu / sqrt(lambda) + phi' * x_tp1); phi is the part of thetas that does not correspond to simultaneous parents
		threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
		numBlocks = dim3((n + threadsPerBlock.x - 1) / threadsPerBlock.x, m);
		batchedComputeNuPlusMu<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const unsigned int*) sp_indices,
				(const DOUBLE**) x_tp1, (const DOUBLE**) thetas, (const DOUBLE**) lambdas, nus_nrepeat_ptr);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after setting nus = nus/sqrt(lambda) + phi' * x_tp1" << ENDL;
	}

	// compute IS weights
	size_t max_i = (n + Gammas_batch_size - 1) / Gammas_batch_size;
	size_t batch_size_last_i = n - (max_i - 1) * Gammas_batch_size;
	size_t IS_index = 0;
	for (size_t i = 0; i < max_i; i++) {
		SYSDEBUG_LOGGER << "   batch " << (i + 1) << " / " << max_i << ENDL;
		IS_index = i * Gammas_batch_size;
		if (i + 1 == max_i) {
			Gammas_batch_size = batch_size_last_i;
		}

		// initialize the info vector to zero
		threadsPerBlock = dim3(THREADS_PER_BLOCK);
		numBlocks = dim3((Gammas_batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
		initVector<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, LU_infos);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after initializing LU_infos vector to zero" << ENDL;

		// initialize Gammas to a zero matrix
		threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
		numBlocks = dim3((m * m + threadsPerBlock.x - 1) / threadsPerBlock.x, Gammas_batch_size);
		batchedInitVector<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, m * m, Gammas);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after initializing Gammas to zero" << ENDL;

		// fill Gammas with the values of the current batch; "Gammas" = (I - Gammas_t)
		threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
		numBlocks = dim3((m * max_p + threadsPerBlock.x - 1) / threadsPerBlock.x, Gammas_batch_size);
		SYSDEBUG_LOGGER << "threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z
				<< ")" << ENDL;
		SYSDEBUG_LOGGER << "numBlocks = (" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << ")" << ENDL;
		fillGammas<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, IS_index, m, max_p, sp_indices, Gammas,
				(const DOUBLE**) thetas);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after filling the Gammas matrix as I - Gamma_t" << ENDL;

		// perform LU factorization of Gammas matrices
		cublasXgetrfBatched(CUBLAS, m, Gammas, m, LU_pivots, LU_infos, Gammas_batch_size);
		//TODO: verify that the LU factorization was successful: check infos

		SYSDEBUG_LOGGER << "after calculating the LU factorization of the Gammas matrix" << ENDL;

		// compute determinants and store into IS_weights
		threadsPerBlock = dim3(THREADS_PER_BLOCK);
		numBlocks = dim3((Gammas_batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
		//DOUBLE* IS_weights_current_pos = (DOUBLE*) ((char*) IS_weights + IS_index * sizeof(DOUBLE)); // = &IS_weights[IS_index]
		//detFromLU<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, m, (const DOUBLE**) Gammas, IS_weights_current_pos);
		detFromLU<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, m, (const DOUBLE**) Gammas, IS_weights,
				IS_index);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after calculating the determinants of the Gammas matrices" << ENDL;

		if (do_forecast) {
			// invert the Gammas matrices
			cublasXgetriBatched(CUBLAS, m, (const DOUBLE**) Gammas, m, LU_pivots, Gammas_inv, m, LU_infos, Gammas_batch_size);

			SYSDEBUG_LOGGER << "after calculating the inverse of the Gammas matrix" << ENDL;

			DOUBLE** nus_current_pos = (DOUBLE**) ((char*) nus_nrepeat_ptr + IS_index * sizeof(DOUBLE*));
			DOUBLE** y_tp1_current_pos = (DOUBLE**) ((char*) y_tp1_nrepeat_ptr + IS_index * sizeof(DOUBLE*));

			cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, plus_one, (const DOUBLE**) Gammas_inv, m,
					(const DOUBLE**) nus_current_pos, m, zero, y_tp1_current_pos, m, Gammas_batch_size);

			SYSDEBUG_LOGGER << "after calculating the forecasts y_tp1" << ENDL;
		}
	}

	// normalize IS_weights
	//cublasXasum(CUBLAS, n, IS_weights, 1, sum_unnormalized_IS_weights);
	sumVector<<<1, 1, 0, stream>>>(n, (const DOUBLE*) IS_weights, sum_unnormalized_IS_weights);
	cudaErrchk(cudaGetLastError());

	threadsPerBlock = dim3(THREADS_PER_BLOCK);
	numBlocks = dim3((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
	scale<<<numBlocks, threadsPerBlock, 0, stream>>>(n, IS_weights, sum_unnormalized_IS_weights,
			SCALE_TRANSFORMATION_INV);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after normalizing the IS_weights" << ENDL; //; the unnormalized weight was: " << *sum_unnormalized_IS_weights << ENDL;

	// compute VB values
	// compute mean_lambdas
	threadsPerBlock = dim3(THREADS_PER_BLOCK);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
	meanScalarAndLog<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, (const DOUBLE*) IS_weights,
			(const DOUBLE**) lambdas, mean_lambdas, mean_log_lambdas);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_lambdas" << ENDL;

	// compute mean_m_t
	threadsPerBlock = dim3(max_p, THREADS_PER_BLOCK / max_p);
	numBlocks = dim3(1, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
	compute_mean_m_t<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const DOUBLE*) IS_weights,
			(const DOUBLE**) lambdas, (const DOUBLE*) mean_lambdas, (const DOUBLE**) thetas, mean_m_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after calculating mean_m_t" << ENDL;

	// compute sqrt(lambda) * (thetas - mean_m_t) and store into thetas
	threadsPerBlock = dim3(1, THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3(m, (n + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	compute_VB_vector1<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const DOUBLE**) lambdas,
			(const DOUBLE**) mean_m_t, thetas);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing sqrt(lambda) * (thetas - mean_m_t)" << ENDL;

	// compute mean_C_t (step 1: V_t)
	threadsPerBlock = dim3(1, THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3(max_p, (m + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	meanBatchedVecVecT<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const unsigned int*) p,
			(const DOUBLE*) IS_weights, (const DOUBLE**) thetas, mean_C_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_C_t" << ENDL;

	// cut off the values of mean_m_t beyond the p-th parameter
	// TODO: check thetas! given that the matrix C_t should already be cut off at p x p, the thetas should be cut off at p ... then this step is unnecessary
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
	batchedCutOffVector<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, (const unsigned int*) p, mean_m_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after cutting off mean_m_t after p" << ENDL;

	// cut off the values of mean_C_t beyond the pxp matrix
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
	batchedCutOffMatrix<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, max_p, (const unsigned int*) p,
			(const unsigned int*) p, mean_C_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after cutting off mean_C_t outside of the p x p matrix" << ENDL;

	// copy mean_C_t into C_t_buffer
	threadsPerBlock = dim3(max_p, THREADS_PER_BLOCK / max_p);
	numBlocks = dim3(max_p, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
	batchedCopy<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p * max_p, (const DOUBLE**) mean_C_t, C_t_buffer);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after copying mean_C_t into C_t_buffer" << ENDL;

	// set diagonal entries outside of the p x p matrix to 1 so that the inverse can be calculated
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);
	batchedSetOutsideSquareMatrixDiagonal<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, (const unsigned int*) p,
			C_t_buffer, (DOUBLE) 1);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after setting the diagonal entries outside of the p x p matrix to 1" << ENDL;

	// compute LU factorization of mean_C_t
	cublasXgetrfBatched(CUBLAS, max_p, C_t_buffer, max_p, INV_pivots, INV_infos, m);

	SYSDEBUG_LOGGER << "after computing the LU factorization of mean_C_t" << ENDL;

	// compute inverse of mean_C_t
	cublasXgetriBatched(CUBLAS, max_p, (const DOUBLE**) C_t_buffer, max_p, INV_pivots, chol_C_t, max_p, INV_infos, m);

	SYSDEBUG_LOGGER << "after computing the inverse of mean_C_t" << ENDL;

	// set the matrix outside of the p x p matrix to 0
	threadsPerBlock = dim3(THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p);
	batchedCutOffMatrix<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, max_p, (const unsigned int*) p,
			(const unsigned int*) p, chol_C_t);
	cudaErrchk(cudaGetLastError());

	// compute mean_Q_t
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_T, CUBLAS_OP_N, 1, max_p, max_p, plus_one, (const DOUBLE**) thetas_nrepeat_ptr,
			max_p, (const DOUBLE**) chol_C_t_nrepeat_ptr, max_p, zero, randoms_nrepeat_ptr, 1, m * n);

	SYSDEBUG_LOGGER << "after computing Q_t (1/2)" << ENDL;

	cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, max_p, plus_one, (const DOUBLE**) randoms_nrepeat_ptr, 1,
			(const DOUBLE**) thetas_nrepeat_ptr, max_p, zero, Q_t_array_ptr, 1, m * n);

	SYSDEBUG_LOGGER << "after computing Q_t (2/2)" << ENDL;

	threadsPerBlock = dim3(THREADS_PER_BLOCK);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
	compute_mean_Q_t<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, (const DOUBLE*) IS_weights, (const DOUBLE**) Q_t,
			mean_Q_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_Q_t" << ENDL;

	// compute mean_n_t
	threadsPerBlock = dim3(THREADS_PER_BLOCK);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
	compute_mean_n_t<<<numBlocks, threadsPerBlock, 0, stream>>>(m, (const unsigned int*) p, (const DOUBLE*) mean_Q_t,
			(const DOUBLE*) mean_lambdas, (const DOUBLE*) mean_log_lambdas, (const DOUBLE*) n_t, mean_n_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_n_t" << ENDL;

	// compute mean_s_t
	threadsPerBlock = dim3(THREADS_PER_BLOCK);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x);
	compute_mean_s_t<<<numBlocks, threadsPerBlock, 0, stream>>>(m, (const unsigned int*) p, (const DOUBLE*) mean_n_t,
			(const DOUBLE*) mean_Q_t, (const DOUBLE*) mean_lambdas, (const DOUBLE*) s_t, mean_s_t);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_s_t" << ENDL;

	// finish computing mean_C_t
	threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
	numBlocks = dim3((m + threadsPerBlock.x - 1) / threadsPerBlock.x, max_p * max_p);
	//batchedScale<<<numBlocks, threadsPerBlock>>>(m, max_p * max_p, mean_C_t, (const DOUBLE*) mean_s_t, SCALE_TRANSFORMATION_NONE);
	batchedScale<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, p, p, mean_C_t, (const DOUBLE*) mean_s_t,
			SCALE_TRANSFORMATION_NONE);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after computing mean_C_t" << ENDL;
}

template<typename DOUBLE> void SGDLM::SGDLM<DOUBLE>::forecast(const DOUBLE* zero, const DOUBLE* plus_one, size_t m,
		size_t max_p, const unsigned int* p, const unsigned int* sp_indices, const DOUBLE** m_t, const DOUBLE** C_t,
		const DOUBLE* n_t, const DOUBLE* s_t, size_t n, size_t Gammas_batch_size, const DOUBLE** const x_tp1,
		DOUBLE** y_tp1_nrepeat_ptr, DOUBLE* data_nus, DOUBLE** nus_nrepeat_ptr, DOUBLE** lambdas, DOUBLE* randoms,
		DOUBLE* randoms_pt2, DOUBLE** randoms_nrepeat_ptr, DOUBLE** Gammas, DOUBLE** Gammas_inv, int* INV_pivots,
		int* INV_infos, DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas, DOUBLE** thetas_nrepeat_ptr,
		cudaStream_t stream, cublasHandle_t CUBLAS, curandGenerator_t CURAND) {
	SYSDEBUG_LOGGER << "SGDLM::forecast()" << ENDL;

	cublasErrchk(cublasSetStream(CUBLAS, stream));

	// sample lambdas ~ Gamma(n_t/2, n_t*s_t/2); thetas ~ N(m_t, C_t/(s_t*lambda_t))
	SGDLM<DOUBLE>::sample_parameters(zero, plus_one, m, max_p, p, m_t, C_t, n_t, s_t, n, lambdas, randoms, randoms_pt2,
			randoms_nrepeat_ptr, chol_C_t, chol_C_t_nrepeat_ptr, thetas, thetas_nrepeat_ptr, stream, CUBLAS, CURAND);

	dim3 threadsPerBlock, numBlocks;

	// sample nus ~ N(0,1)
	curandGenerateNormalX(CURAND, data_nus, n * m, 0, 1);

	SYSDEBUG_LOGGER << "after sampling nus ~ N(0,1)" << ENDL;

	// compute nu = (nu / sqrt(lambda) + phi' * x_tp1); phi is the part of thetas that does not correspond to simultaneous parents
	threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
	numBlocks = dim3((n + threadsPerBlock.x - 1) / threadsPerBlock.x, m);
	batchedComputeNuPlusMu<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const unsigned int*) sp_indices,
			(const DOUBLE**) x_tp1, (const DOUBLE**) thetas, (const DOUBLE**) lambdas, nus_nrepeat_ptr);
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after setting nus = nus/sqrt(lambda) + phi' * x_tp1" << ENDL;

	// compute forecasts
	size_t max_i = (n + Gammas_batch_size - 1) / Gammas_batch_size;
	size_t batch_size_last_i = n - (max_i - 1) * Gammas_batch_size;
	size_t SIM_index = 0;
	for (size_t i = 0; i < max_i; i++) {
		SYSDEBUG_LOGGER << "   batch " << (i + 1) << " / " << max_i << ENDL;
		SIM_index = i * Gammas_batch_size;
		if (i + 1 == max_i) {
			Gammas_batch_size = batch_size_last_i;
		}

		// initialize Gammas to a zero matrix
		threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
		numBlocks = dim3((m * m + threadsPerBlock.x - 1) / threadsPerBlock.x, Gammas_batch_size);
		batchedInitVector<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, m * m, Gammas);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after initializing Gammas to zero" << ENDL;

		// fill Gammas with the values of the current batch; "Gammas" = (I - Gammas_t)
		threadsPerBlock = dim3(THREADS_PER_BLOCK, 1);
		numBlocks = dim3((m * max_p + threadsPerBlock.x - 1) / threadsPerBlock.x, Gammas_batch_size);
		SYSDEBUG_LOGGER << "threadsPerBlock = (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z
				<< ")" << ENDL;
		SYSDEBUG_LOGGER << "numBlocks = (" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << ")" << ENDL;
		fillGammas<<<numBlocks, threadsPerBlock, 0, stream>>>(Gammas_batch_size, SIM_index, m, max_p, sp_indices,
				Gammas, (const DOUBLE**) thetas);
		cudaErrchk(cudaGetLastError());

		SYSDEBUG_LOGGER << "after filling the Gammas matrix as I - Gamma_t" << ENDL;

		// perform LU factorization of Gammas matrices
		cublasXgetrfBatched(CUBLAS, m, Gammas, m, INV_pivots, INV_infos, Gammas_batch_size);
		//TODO: verify that the LU factorization was successful: check infos

		SYSDEBUG_LOGGER << "after calculating the LU factorization of the Gammas matrix" << ENDL;

		// invert the Gammas matrices
		cublasXgetriBatched(CUBLAS, m, (const DOUBLE**) Gammas, m, INV_pivots, Gammas_inv, m, INV_infos, Gammas_batch_size);

		SYSDEBUG_LOGGER << "after calculating the inverse of the Gammas matrix" << ENDL;

		DOUBLE** nus_current_pos = (DOUBLE**) ((char*) nus_nrepeat_ptr + SIM_index * sizeof(DOUBLE*));
		DOUBLE** y_tp1_current_pos = (DOUBLE**) ((char*) y_tp1_nrepeat_ptr + SIM_index * sizeof(DOUBLE*));

		cublasXgemmBatched(CUBLAS, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, m, plus_one, (const DOUBLE**) Gammas_inv, m,
				(const DOUBLE**) nus_current_pos, m, zero, y_tp1_current_pos, m, Gammas_batch_size);

		SYSDEBUG_LOGGER << "after calculating the forecasts y_tp1" << ENDL;
	}
}

template<typename DOUBLE> void SGDLM::SGDLM<DOUBLE>::sample_parameters(const DOUBLE* zero, const DOUBLE* plus_one, size_t m,
		size_t max_p, const unsigned int* p, const DOUBLE** m_t, const DOUBLE** C_t, const DOUBLE* n_t,
		const DOUBLE* s_t, size_t n, DOUBLE** lambdas, DOUBLE* randoms, DOUBLE* randoms_pt2,
		DOUBLE** randoms_nrepeat_ptr, DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas,
		DOUBLE** thetas_nrepeat_ptr, cudaStream_t stream, cublasHandle_t CUBLAS, curandGenerator_t CURAND) {
	SYSDEBUG_LOGGER << "SGDLM::sample_parameters()" << ENDL;

	cublasErrchk(cublasSetStream(CUBLAS, stream));

	// sample lambda ~ Gamma(n_t/2, n_t*s_t/2)
	// point to memory from randoms to use as cache; max_p must be greater than or equal to 4 for this to work with GAMMA_LOOPS = 2
	sampleGamma2(CURAND, m, n, n_t, s_t, randoms, randoms_pt2, lambdas, stream);

	SYSDEBUG_LOGGER << "after sampling lambdas ~ Gamma(n_t/2, n_t*s_t/2)" << ENDL;

	// sample theta ~ N(0,1)
	curandGenerateNormalX(CURAND, randoms, n * m * max_p, 0, 1);

	SYSDEBUG_LOGGER << "after sampling randoms ~ N(0,1)" << ENDL;

	// Cholesky of C_t
	dim3 threadsPerBlock(1, THREADS_PER_BLOCK);
	dim3 numBlocks(1, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
	cholesky<<<numBlocks, threadsPerBlock, 0, stream>>>(m, max_p, max_p, (const unsigned int*) p, (const DOUBLE**) C_t,
			chol_C_t);
	cudaErrchk(cudaGetLastError());
	//cudaErrchk(cudaDeviceSynchronize());

	SYSDEBUG_LOGGER << "after calculating the cholesky factorization of C_t" << ENDL;

	// make MVN from iid standard normals: theta_j ~ N(m_j, C_j)
	cublasXgemmBatched(CUBLAS, CUBLAS_OP_T, CUBLAS_OP_N, max_p, 1, max_p, plus_one,
			(const DOUBLE**) chol_C_t_nrepeat_ptr, max_p, (const DOUBLE**) randoms_nrepeat_ptr, max_p, zero,
			thetas_nrepeat_ptr, max_p, m * n);
	//cudaErrchk(cudaDeviceSynchronize());

	SYSDEBUG_LOGGER << "after thetas = chol(C_t) * randoms" << ENDL;

	threadsPerBlock = dim3(1, THREADS_PER_BLOCK / max_p, max_p);
	numBlocks = dim3(n, (m + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	makeVB_MVN<<<numBlocks, threadsPerBlock, 0, stream>>>(n, m, max_p, (const DOUBLE**) m_t, (const DOUBLE**) lambdas,
			(const DOUBLE*) s_t, thetas); // divide by sqrt(s_t * lambda_t), then add m_t as mean value
	cudaErrchk(cudaGetLastError());

	SYSDEBUG_LOGGER << "after thetas += m_t" << ENDL;
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
template class SGDLM::SGDLM<DOUBLETYPE>;
