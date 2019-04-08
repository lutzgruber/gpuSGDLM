/*
 * SGDLMHostWrapperImpl.cuh
 *
 *  Created on: Jul 26, 2015
 *      Author: lutz
 */

#ifndef SGDLMHOSTWRAPPERIMPL_CUH_
#define SGDLMHOSTWRAPPERIMPL_CUH_

#include "memory_manager_GPU.cuh"
#include "simPointers.cuh"
#include "SGDLM/HostWrapper.hpp"

namespace SGDLM {

template<typename DOUBLE>
class HostWrapperImpl: public HostWrapper<DOUBLE> {
public:
	HostWrapperImpl(std::size_t no_gpus);

	~HostWrapperImpl();

	std::size_t getNoSeries() const;

	std::size_t getMaxP() const;

	bool getEvolutionMatrixConfiguration() const;

	void initMemory(std::size_t m, std::size_t max_p); //, const DOUBLE* host_data_m_t, const DOUBLE* host_data_C_t, const DOUBLE* host_data_n_t, const DOUBLE* host_data_s_t, const DOUBLE* host_data_beta, const DOUBLE* host_data_delta, const unsigned int* host_data_p, const unsigned int* host_data_sp_indices);

	void clearMemory();

	void manageEvoMemory(bool use_state_evolution_matrix);

	std::size_t getNSim() const;

	std::size_t getNSimBatch() const;

	void initSimMemory(std::size_t nsim, std::size_t nsim_batch);

	void clearSimMemory();

	bool isPrior() const;

	void isPrior(bool is_prior);

	void getParameters(DOUBLE* host_data_m_t, DOUBLE* host_data_C_t, DOUBLE* host_data_n_t,
			DOUBLE* host_data_s_t) const;

	void setParameters(const DOUBLE* host_data_m_t, const DOUBLE* host_data_C_t, const DOUBLE* host_data_n_t,
			const DOUBLE* host_data_s_t);

	void getDiscountFactors(DOUBLE* host_data_beta, DOUBLE* host_data_delta) const;

	void setDiscountFactors(const DOUBLE* host_data_beta, const DOUBLE* host_data_delta);

	void getEvolutionMatrix(DOUBLE* host_data_G_t) const;

	void setEvolutionMatrix(const DOUBLE* host_data_G_t);

	void getParentalSets(unsigned int* host_data_p, unsigned int* host_data_sp_indices) const;

	void setParentalSets(const unsigned int* host_data_p, const unsigned int* host_data_sp_indices);

	void computePrior();

	void computePrior(const DOUBLE* host_data_G_tp1);

	void computeForecast(DOUBLE* host_data_y_tp1, const DOUBLE* host_data_x_tp1);

	void computePosterior(const DOUBLE* host_data_y_t, const DOUBLE* host_data_F_t);

	void computeVBPosterior(DOUBLE* host_data_mean_m_t, DOUBLE* host_data_mean_C_t, DOUBLE* host_data_mean_n_t,
			DOUBLE* host_data_mean_s_t, DOUBLE* host_data_IS_weights, DOUBLE* host_sum_det_weights);
private:
	bool memory_initialized;
	bool sim_memory_initialized;
	bool evo_memory_initialized;
	bool is_prior;
	bool use_state_evolution_matrix;

	int i;
	int no_gpus;
	int main_gpu;

	std::size_t m;
	std::size_t max_p;
	std::size_t nsim;
	std::size_t nsim_batch;

	simPointers<DOUBLE, memory_manager_GPU> simP[MAX_NO_GPUS];

	bool checkInitialized() const;

	bool checkSimInitialized() const;

	bool checkUseStateEvolutionMatrix() const;

	bool checkPrior(bool is_prior) const;

	void allocate_C_t_memory();
};

}

#endif /* SGDLMHOSTWRAPPERIMPL_CUH_ */
