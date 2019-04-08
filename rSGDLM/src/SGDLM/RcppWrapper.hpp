/*
 * RcppWrapper.cuh
 *
 *  Created on: Jul 22, 2015
 *      Author: lutz
 */

#ifndef RCPPWRAPPER_CUH_
#define RCPPWRAPPER_CUH_

#ifdef USE_RCPP

#include <Rcpp.h>

#include <vector>

#include "memory_manager_RCPP.hpp"

#include "SGDLM/HostWrapper.hpp"

#include "logger.hpp"

namespace SGDLM {

template<typename DOUBLE>
class RcppWrapper {
public:
	RcppWrapper();

	RcppWrapper(std::size_t no_gpus);

	~RcppWrapper();

	void clearGPUAllMemory();

	void clearGPUSimMemory();

	Rcpp::List getSimultaneousParents() const;

	void setSimultaneousParents(const std::vector<unsigned int>& p, const Rcpp::IntegerVector& sp);

	Rcpp::List getDiscountFactors() const;

	void setDiscountFactors(const std::vector<DOUBLE>& beta, const Rcpp::NumericVector& delta);

	Rcpp::List getParameters() const;

	void setPriorParameters(const Rcpp::NumericVector& m_t, const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t);

	void setPosteriorParameters(const Rcpp::NumericVector& m_t, const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t);

	void computePosterior(const std::vector<DOUBLE>& y_t, const Rcpp::NumericVector& F_t);

	Rcpp::List computeVBPosterior(std::size_t nsim, std::size_t nsim_batch);

	void computePrior();

	void computeEvoPrior(const Rcpp::NumericVector& G_tp1);

	Rcpp::NumericVector computeForecast(std::size_t nsim, std::size_t nsim_batch, const Rcpp::NumericVector& F_tp1);

	std::size_t getLogLevel() const;

	void setLogLevel(std::size_t loglevel);
private:
	SGDLM::HostWrapper<DOUBLE>* wrapper;

	void setParameters(const Rcpp::NumericVector& m_t, const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t, bool is_prior);
};

}

#endif


#endif /* RCPPWRAPPER_CUH_ */
