/*
 * memory_manager_RCPP.hpp
 *
 *  Created on: Jul 22, 2015
 *      Author: lutz
 */

#ifndef MEMORY_MANAGER_RCPP_HPP_
#define MEMORY_MANAGER_RCPP_HPP_

#ifdef USE_RCPP

#include <Rcpp.h>

#include "logger.hpp"

#include "memory_manager.hpp"

#include <vector>

class memory_manager_RCPP: public memory_manager {
public:
	memory_manager_RCPP() :
			memory_manager() {
		SYSDEBUG_LOGGER << "memory_manager_RCPP::memory_manager_RCPP()" << ENDL;
	}

	virtual ~memory_manager_RCPP() {
		SYSDEBUG_LOGGER << "memory_manager_RCPP::~memory_manager_RCPP()" << ENDL;
	}

	template<typename T, typename vector_type> T* getData(const vector_type& vec) {
		SYSDEBUG_LOGGER << "memory_manager_RCPP::getData()" << ENDL;

		T* host_data_pointer = host_alloc_vec<T>(vec.size());

		T* running_pointer = host_data_pointer;
		for (typename vector_type::const_iterator it = vec.begin(); it != vec.end(); it++, running_pointer++) {
			SYSDEBUG_LOGGER << "*it = " << *it << ENDL;
			*running_pointer = (T) *it;
		}

		return host_data_pointer;
	}

	template<typename T, typename vector_type> static vector_type getRcppVector(const T* host_data_ptr,
			std::size_t no_elements) {
		return getRcppVector<T, vector_type>(host_data_ptr, 1, &no_elements);
	}

	template<typename T, typename vector_type> static vector_type getRcppVector(const T* host_data_ptr,
			std::size_t ndims, const std::size_t* dims) {
		std::vector<std::size_t> dim(ndims);

		for (std::vector<std::size_t>::iterator it = dim.begin(); it != dim.end(); it++) {
			*it = *dims;
			dims++;
		}

		return getRcppVector<T, vector_type>(host_data_ptr, dim);
	}

	template<typename T, typename vector_type> static vector_type getRcppVector(const T* host_data_ptr,
			std::vector<std::size_t>& dim) {
		Rcpp::Dimension vec_dim = Rcpp::wrap(dim);

		vector_type vec(vec_dim);

		for (typename vector_type::iterator it = vec.begin(); it != vec.end(); it++) {
			*it = (T) *host_data_ptr;
			host_data_ptr++;
		}

		return vec;
	}

	template<typename T> static std::vector<T> getSTDVector(const T* host_data_ptr, std::size_t no_elements) {
		std::vector<T> vec(no_elements);

		for (typename std::vector<T>::iterator it = vec.begin(); it != vec.end(); it++) {
			*it = *host_data_ptr;
			host_data_ptr++;
		}

		return vec;
	}

	template<typename T> static std::vector<T> getSTDVector(const T* host_data_ptr, std::size_t ndims,
			const std::size_t* dims) {
		std::size_t no_elements = ndims > 0 ? 1 : 0;
		for (std::size_t i = 0; i < ndims; i++) {
			no_elements *= dims[i];
		}

		return getSTDVector<T>(host_data_ptr, no_elements);
	}

	template<typename T> static std::vector<T> getSTDVector(const T* host_data_ptr, std::vector<std::size_t>& dim) {
		std::size_t no_elements = dim.size() > 0 ? 1 : 0;
		for (std::size_t i = 0; i < dim.size(); i++) {
			no_elements *= dim[i];
		}

		return getSTDVector<T>(host_data_ptr, no_elements);
	}
};

#endif

#endif /* MEMORY_MANAGER_RCPP_HPP_ */
