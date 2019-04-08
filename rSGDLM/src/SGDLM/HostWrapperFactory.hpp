/*
 * SGDLMHostWrapperFactory.hpp
 *
 *  Created on: Jul 27, 2015
 *      Author: lutz
 */

#ifndef SGDLMHOSTWRAPPERFACTORY_HPP_
#define SGDLMHOSTWRAPPERFACTORY_HPP_

#include "SGDLM/HostWrapper.hpp"

namespace SGDLM {

template<typename DOUBLE> class HostWrapperFactory {
public:
	static SGDLM::HostWrapper<DOUBLE>* create(std::size_t no_gpus);

	static SGDLM::HostWrapper<DOUBLE>* create(std::size_t no_gpus, std::size_t m, std::size_t max_p);

	static SGDLM::HostWrapper<DOUBLE>* create(std::size_t no_gpus, std::size_t m, std::size_t max_p, bool use_state_evolution_matrix);
};

}


#endif /* SGDLMHOSTWRAPPERFACTORY_HPP_ */
