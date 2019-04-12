/*
 * R_main.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: lutz
 */

#ifdef USE_RCPP

#include <Rcpp.h>

#include "SGDLM/RcppWrapper.hpp"

typedef SGDLM::RcppWrapper<DOUBLETYPE> wrapper_type;

RCPP_MODULE(gpuSGDLM) {
	Rcpp::class_<wrapper_type>("SGDLM")
	.constructor()
	.constructor<std::size_t>()
	.method("getSimultaneousParents", &wrapper_type::getSimultaneousParents)
	.method("setSimultaneousParents", &wrapper_type::setSimultaneousParents)
	.method("getDiscountFactors", &wrapper_type::getDiscountFactors)
	.method("setDiscountFactors", &wrapper_type::setDiscountFactors)
	.method("getParameters", &wrapper_type::getParameters)
	.method("setPriorParameters", &wrapper_type::setPriorParameters)
	.method("setPosteriorParameters", &wrapper_type::setPosteriorParameters)
	.method("computePosterior", &wrapper_type::computePosterior)
	.method("computeVBPosterior", &wrapper_type::computeVBPosterior)
	.method("computePrior", &wrapper_type::computePrior)
	.method("computeEvoPrior", &wrapper_type::computeEvoPrior)
	.method("computeForecast", &wrapper_type::computeForecast)
	.method("getLogLevel", &wrapper_type::getLogLevel)
	.method("setLogLevel", &wrapper_type::setLogLevel)
	.method("clearGPUMemory", &wrapper_type::clearGPUAllMemory)
	.method("clearSimulationMemory", &wrapper_type::clearGPUSimMemory)
	;
}


#endif
