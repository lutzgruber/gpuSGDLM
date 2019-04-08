/*
 * SGDLMHostWrapper.hpp
 *
 *  Created on: Jul 27, 2015
 *      Author: lutz
 */

#ifndef SGDLMHOSTWRAPPER_HPP_
#define SGDLMHOSTWRAPPER_HPP_

namespace SGDLM {

template<typename DOUBLE>
class HostWrapper {
public:
	virtual ~HostWrapper() {

	}

	virtual std::size_t getNoSeries() const = 0;

	virtual std::size_t getMaxP() const = 0;

	virtual bool getEvolutionMatrixConfiguration() const = 0;

	virtual void initMemory(std::size_t m, std::size_t max_p) = 0; //, const DOUBLE* host_data_m_t, const DOUBLE* host_data_C_t, const DOUBLE* host_data_n_t, const DOUBLE* host_data_s_t, const DOUBLE* host_data_beta, const DOUBLE* host_data_delta, const unsigned int* host_data_p, const unsigned int* host_data_sp_indices);

	virtual void clearMemory() = 0;

	virtual void manageEvoMemory(bool use_state_evolution_matrix) = 0;

	virtual std::size_t getNSim() const = 0;

	virtual std::size_t getNSimBatch() const = 0;

	virtual void initSimMemory(std::size_t nsim, std::size_t nsim_batch) = 0;

	virtual void clearSimMemory() = 0;

	virtual bool isPrior() const = 0;

	virtual void isPrior(bool is_prior) = 0;

	virtual void getParameters(DOUBLE* host_data_m_t, DOUBLE* host_data_C_t, DOUBLE* host_data_n_t,
			DOUBLE* host_data_s_t) const = 0;

	virtual void setParameters(const DOUBLE* host_data_m_t, const DOUBLE* host_data_C_t, const DOUBLE* host_data_n_t,
			const DOUBLE* host_data_s_t) = 0;

	virtual void getDiscountFactors(DOUBLE* host_data_beta, DOUBLE* host_data_delta) const = 0;

	virtual void setDiscountFactors(const DOUBLE* host_data_beta, const DOUBLE* host_data_delta) = 0;

	virtual void getEvolutionMatrix(DOUBLE* host_data_G_t) const = 0;

	virtual void setEvolutionMatrix(const DOUBLE* host_data_G_t) = 0;

	virtual void getParentalSets(unsigned int* host_data_p, unsigned int* host_data_sp_indices) const = 0;

	virtual void setParentalSets(const unsigned int* host_data_p, const unsigned int* host_data_sp_indices) = 0;

	virtual void computePrior() = 0;

	virtual void computePrior(const DOUBLE* host_data_G_tp1) = 0;

	virtual void computeForecast(DOUBLE* host_data_ytp1, const DOUBLE* host_data_x_tp1) = 0;

	virtual void computePosterior(const DOUBLE* host_data_y_t, const DOUBLE* host_data_F_t) = 0;

	virtual void computeVBPosterior(DOUBLE* host_data_mean_m_t, DOUBLE* host_data_mean_C_t, DOUBLE* host_data_mean_n_t,
			DOUBLE* host_data_mean_s_t, DOUBLE* host_data_IS_weights, DOUBLE* host_sum_det_weights) = 0;
};

}

#endif /* SGDLMHOSTWRAPPER_HPP_ */
