/*
 * memory_manager.hpp
 *
 *  Created on: Jul 28, 2015
 *      Author: lutz
 */

#ifndef MEMORY_MANAGER_HPP_
#define MEMORY_MANAGER_HPP_

#include "logger.hpp"

#include <vector>

class memory_manager {
public:
	memory_manager() :
			free_array() {
		SYSDEBUG_LOGGER << "memory_manager::memory_manager()" << ENDL;
	}

	virtual ~memory_manager() {
		SYSDEBUG_LOGGER << "memory_manager::~memory_manager()" << ENDL;

		clear();
	}

	template<typename T> inline T _free(T ptr) {
		SYSDEBUG_LOGGER << "memory_manager::_free()" << ENDL;

		free_array.push_back(ptr);

		return ptr;
	}

	template<typename T> inline T* host_alloc(size_t memory_size) {
		SYSDEBUG_LOGGER << "memory_manager::host_alloc()" << ENDL;

		T* host_ptr = (T*) malloc(memory_size);

		free_array.push_back(host_ptr);

		return host_ptr;
	}

	template<typename T> inline T* host_alloc_vec(size_t no_elements, bool initialize_to_zero = false) {
		SYSDEBUG_LOGGER << "memory_manager::host_alloc()" << ENDL;

		T* host_ptr = (T*) malloc(no_elements * sizeof(T));

		if (initialize_to_zero) {
			for (std::size_t i = 0; i < no_elements; i++) {
				host_ptr[i] = 0;
			}
		}

		free_array.push_back(host_ptr);

		return host_ptr;
	}

	template<typename T> inline T* host_alloc(size_t ndims, const size_t* dims) {
		SYSDEBUG_LOGGER << "memory_manager::host_alloc()" << ENDL;

		size_t memory_size = sizeof(T);
		for (size_t i = 0; i < ndims; i++) {
			memory_size *= dims[i];
		}

		return host_alloc<T>(memory_size);
	}

	void clear() {
		SYSDEBUG_LOGGER << "memory_manager::clear()" << ENDL;

		/*startCuda(gpu_index);

		 if (stream != NULL) {
		 cudaErrchk(cudaStreamSynchronize(stream));
		 }*/

		for (std::vector<void*>::reverse_iterator it = free_array.rbegin(); it != free_array.rend(); ++it) {
			free(*it);
			*it = NULL;
		}
		free_array.clear();
		SYSDEBUG_LOGGER << "...finished free(.)" << ENDL;

		this->clear_derived();
	}
private:
	std::vector<void*> free_array;

	virtual void clear_derived() {
		SYSDEBUG_LOGGER << "memory_manager::clear_derived()" << ENDL;
	}
};

#endif /* MEMORY_MANAGER_HPP_ */
