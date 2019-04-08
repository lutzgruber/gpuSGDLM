#include "SGDLM/HostWrapperFactory.hpp"

#include "SGDLM/HostWrapperImpl.cuh"


template<typename DOUBLE> SGDLM::HostWrapper<DOUBLE>* SGDLM::HostWrapperFactory<DOUBLE>::create(std::size_t no_gpus) {
	SGDLM::HostWrapper<DOUBLE>* wrapper = new HostWrapperImpl<DOUBLE>(no_gpus);

	return wrapper;
}

template<typename DOUBLE> SGDLM::HostWrapper<DOUBLE>* SGDLM::HostWrapperFactory<DOUBLE>::create(std::size_t no_gpus, std::size_t m, std::size_t max_p) {
	SGDLM::HostWrapper<DOUBLE>* wrapper = new HostWrapperImpl<DOUBLE>(no_gpus);

	wrapper->initMemory(m, max_p);

	return wrapper;
}

template<typename DOUBLE> SGDLM::HostWrapper<DOUBLE>* SGDLM::HostWrapperFactory<DOUBLE>::create(std::size_t no_gpus, std::size_t m, std::size_t max_p, bool use_state_evolution_matrix) {
	SGDLM::HostWrapper<DOUBLE>* wrapper = new HostWrapperImpl<DOUBLE>(no_gpus);

	wrapper->initMemory(m, max_p);
	wrapper->manageEvoMemory(use_state_evolution_matrix);

	return wrapper;
}

/*
 *
 *
 *
 *
 *
 *
 */

//explicit instantiation
template class SGDLM::HostWrapperFactory<DOUBLETYPE>;
