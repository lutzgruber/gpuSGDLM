/*
 * memory_manager_GPU.cuh
 *
 *  Created on: Dec 8, 2013
 *      Author: lutz
 */

#ifndef memory_manager_GPU_CUH_
#define memory_manager_GPU_CUH_

#include <cuda_runtime.h>
#include "loggerGPU.cuh"
#include "cuda_manager.cuh"
#include "memory_manager.hpp"

#include <vector>

class memory_manager_GPU: public memory_manager {
public:
	memory_manager_GPU(size_t pGPUIndex, cudaStream_t pStream) : memory_manager(),
			cudafree_array(), cudahostfree_array(), stream(pStream), gpu_index(pGPUIndex) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::memory_manager_GPU()" << ENDL;
	}

	memory_manager_GPU() :
			memory_manager(), cudafree_array(), cudahostfree_array(), stream(NULL), gpu_index(0) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::memory_manager_GPU()" << ENDL;
	}

	virtual ~memory_manager_GPU() {
		SYSDEBUG_LOGGER << "memory_manager_GPU::~memory_manager_GPU()" << ENDL;

		clear_derived();
	}

	template<typename T> static inline T* dereferencePtrArray(T** dev_ptr_array) {
		T** ptr = (T**) malloc(sizeof(T*));

		cudaErrchk(cudaMemcpy(ptr, dev_ptr_array, sizeof(T*), cudaMemcpyDeviceToHost));

		T* ptr2 = ptr[0];

		free(ptr);

		return ptr2;
	}

	// host_data_ptr must already be initialized!!
	template<typename T> inline void cpyToHost(const T* device_data_ptr, T* host_data_ptr, size_t no_elements) const {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToHost()" << ENDL;

		//size_t memory_size = sizeof(T) * no_elements;
		//
		//cudaErrchk(cudaMemcpyAsync(host_data_ptr, device_data_ptr, memory_size, cudaMemcpyDeviceToHost, this->stream));

		cpyToHost<T>(device_data_ptr, host_data_ptr, no_elements, this->stream);
	}

	// host_data_ptr must already be initialized!!
	template<typename T> inline static void cpyToHost(const T* device_data_ptr, T* host_data_ptr, size_t no_elements, cudaStream_t stream) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToHost()" << ENDL;

		size_t memory_size = sizeof(T) * no_elements;

		cudaErrchk(cudaMemcpyAsync(host_data_ptr, device_data_ptr, memory_size, cudaMemcpyDeviceToHost, stream));
	}

	// device_data_ptr must already be initialized!!
	template<typename T> inline void cpyToDevice(T* device_data_ptr, const T* host_data_ptr, size_t no_elements) const {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToDevice()" << ENDL;

		//size_t memory_size = sizeof(T) * no_elements;
		//
		//cudaErrchk(cudaMemcpyAsync(device_data_pointer, host_data_ptr, memory_size, cudaMemcpyHostToDevice, this->stream));

		cpyToDevice<T>(device_data_ptr, host_data_ptr, no_elements, this->stream);
	}

	// device_data_ptr must already be initialized
	template<typename T> inline static void cpyToDevice(T* device_data_ptr, const T* host_data_ptr, size_t no_elements, cudaStream_t stream) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToDevice()" << ENDL;

		size_t memory_size = sizeof(T) * no_elements;

		cudaErrchk(cudaMemcpyAsync(device_data_ptr, host_data_ptr, memory_size, cudaMemcpyHostToDevice, stream));
	}

	template<typename T> inline T* cpyToDevice(const T* host_data_ptr, size_t no_elements, unsigned int flags = UINT_MAX) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToDevice()" << ENDL;

		size_t memory_size = sizeof(T) * no_elements;

		T* cpu_gpu_pointer;

		if (flags == UINT_MAX) {
			cpu_gpu_pointer = device_alloc<T>(memory_size);
		} else {
			cpu_gpu_pointer = host_device_alloc<T>(memory_size, flags);
		}

		cudaErrchk(cudaMemcpyAsync(cpu_gpu_pointer, host_data_ptr, memory_size, cudaMemcpyHostToDevice, this->stream));

		//cudaErrchk(cudaDeviceSynchronize());

		return cpu_gpu_pointer;
	}

	template<typename T> inline void cpyToDeviceAsPtrArray(const T* dev_data_ptr, size_t no_matrices,
			size_t matrix_size, T**& host_dev_array_ptr, bool host_device = false) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToDeviceAsPtrArray()" << ENDL;

		size_t memory_size = no_matrices * sizeof(T*);

		if (host_device) {
			host_dev_array_ptr = host_device_alloc<T*>(memory_size);
			for (size_t i = 0; i < no_matrices; i++) {
				host_dev_array_ptr[i] = (T*) ((char*) dev_data_ptr + i * matrix_size * sizeof(T));
			}
		} else {
			T** host_array_ptr = (T**) malloc(memory_size);
			for (size_t i = 0; i < no_matrices; i++) {
				host_array_ptr[i] = (T*) ((char*) dev_data_ptr + i * matrix_size * sizeof(T));
			}

			host_dev_array_ptr = device_alloc<T*>(memory_size);
			cudaErrchk(cudaMemcpyAsync(host_dev_array_ptr, host_array_ptr, memory_size, cudaMemcpyHostToDevice, stream));

			free(host_array_ptr);
		}
	}

	template<typename T> inline void cpyToDeviceAsPtrArrayByCol(const T* dev_data_ptr, size_t no_matrices,
			size_t no_vectors_per_matrix, size_t vector_length, T**& host_dev_array_ptr, bool host_device = false) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::getPtrArrayByCol()" << ENDL;

		size_t memory_size = no_matrices * no_vectors_per_matrix * sizeof(T*);

		if (host_device) {
			host_dev_array_ptr = host_device_alloc<T*>(memory_size);
			for (size_t i = 0; i < no_matrices * no_vectors_per_matrix; i++) {
				host_dev_array_ptr[i] = (T*) ((char*) dev_data_ptr + i * vector_length * sizeof(T));
			}
		} else {
			T** host_array_ptr = (T**) malloc(memory_size);
			for (size_t i = 0; i < no_matrices * no_vectors_per_matrix; i++) {
				host_array_ptr[i] = (T*) ((char*) dev_data_ptr + i * vector_length * sizeof(T));
			}

			host_dev_array_ptr = device_alloc<T*>(memory_size);
			cudaErrchk(cudaMemcpyAsync(host_dev_array_ptr, host_array_ptr, memory_size, cudaMemcpyHostToDevice, stream));

			free(host_array_ptr);
		}
	}

	template<typename T> inline void cpyToDeviceAsPtrArrayRepeatByBatch(const T* dev_data_ptr, size_t no_matrices,
			size_t matrix_size, size_t no_repeats, T**& host_dev_array_ptr, bool host_device = false) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::cpyToDeviceAsPtrArrayRepeatByBatch()" << ENDL;

		size_t memory_size = no_matrices * no_repeats * sizeof(T*);

		if (host_device) {
			host_dev_array_ptr = host_device_alloc<T*>(memory_size);
			for (size_t i = 0; i < no_matrices; i++) {
				for (size_t repeat = 0; repeat < no_repeats; repeat++) {
					host_dev_array_ptr[repeat * no_matrices + i] = (T*) ((char*) dev_data_ptr
							+ i * matrix_size * sizeof(T));
				}
			}
		} else {
			T** host_array_ptr = (T**) malloc(memory_size);
			for (size_t i = 0; i < no_matrices; i++) {
				for (size_t repeat = 0; repeat < no_repeats; repeat++) {
					host_array_ptr[repeat * no_matrices + i] =
							(T*) ((char*) dev_data_ptr + i * matrix_size * sizeof(T));
				}
			}

			host_dev_array_ptr = device_alloc<T*>(memory_size);
			cudaErrchk(cudaMemcpyAsync(host_dev_array_ptr, host_array_ptr, memory_size, cudaMemcpyHostToDevice, stream));

			free(host_array_ptr);

		}
	}

	template<typename T> inline T _cudaFree(T ptr) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::_cudaFree()" << ENDL;

		cudafree_array.push_back(ptr);

		return ptr;
	}

	template<typename T> inline T _cudaFreeHost(T ptr) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::_cudaFreeHost()" << ENDL;

		cudahostfree_array.push_back(ptr);

		return ptr;
	}

	template<typename T> inline void _cudaMalloc(T*& ptr, size_t size) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::_cudaMalloc()" << ENDL;

		cudaErrchk(cudaMalloc(&ptr, size));

		cudafree_array.push_back(ptr);
	}

	template<typename T> inline void _cudaHostAlloc(T*& ptr, size_t size,
			unsigned int flags = cudaHostAllocPortable & cudaHostAllocMapped) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::_cudaHostAlloc()" << ENDL;

		cudaErrchk(cudaHostAlloc<T>(&ptr, size, flags));

		cudahostfree_array.push_back(ptr);
	}

	template<typename T> inline T* host_device_alloc(size_t memory_size,
			unsigned int flags = cudaHostAllocPortable & cudaHostAllocMapped) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::host_device_alloc()" << ENDL;

		T* host_device_ptr = NULL;

		_cudaHostAlloc<T>(host_device_ptr, memory_size, flags);

		return host_device_ptr;
	}

	template<typename T> inline T* host_device_alloc_vec(size_t no_elements,
			unsigned int flags = cudaHostAllocPortable & cudaHostAllocMapped) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::host_device_alloc()" << ENDL;

		T* host_device_ptr = NULL;

		_cudaHostAlloc<T>(host_device_ptr, no_elements * sizeof(T), flags);

		return host_device_ptr;
	}

	template<typename T> inline T* host_device_alloc(size_t ndims, const size_t* dims,
			unsigned int flags = cudaHostAllocPortable & cudaHostAllocMapped) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::host_device_alloc()" << ENDL;

		T* host_device_ptr = NULL;

		size_t memory_size = sizeof(T);
		for (size_t i = 0; i < ndims; i++) {
			memory_size *= dims[i];
		}

		_cudaHostAlloc<T>(host_device_ptr, memory_size);

		return host_device_ptr;
	}

	template<typename T> inline T* device_alloc(size_t memory_size) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::device_alloc()" << ENDL;

		T* device_ptr = NULL;

		_cudaMalloc<T>(device_ptr, memory_size);

		return device_ptr;
	}

	template<typename T> inline T* device_alloc_vec(size_t no_elements) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::device_alloc()" << ENDL;

		T* device_ptr = NULL;

		_cudaMalloc<T>(device_ptr, no_elements * sizeof(T));

		return device_ptr;
	}

	template<typename T> inline T* device_alloc(size_t ndims, const size_t* dims) {
		SYSDEBUG_LOGGER << "memory_manager_GPU::device_alloc()" << ENDL;

		T* device_ptr = NULL;

		size_t memory_size = sizeof(T);
		for (size_t i = 0; i < ndims; i++) {
			memory_size *= dims[i];
		}

		_cudaMalloc<T>(device_ptr, memory_size);

		return device_ptr;
	}

private:
	std::vector<void*> cudafree_array;
	std::vector<void*> cudahostfree_array;
	size_t gpu_index;
	cudaStream_t stream;

	virtual void clear_derived() {
		SYSDEBUG_LOGGER << "memory_manager_GPU::clear_derived()" << ENDL;

		for (std::vector<void*>::reverse_iterator it = cudafree_array.rbegin(); it != cudafree_array.rend(); ++it) {
			cudaErrchk(cudaFree(*it));
			*it = NULL;
		}
		cudafree_array.clear();
		SYSDEBUG_LOGGER << "...finished cudaFree(.)" << ENDL;

		for (std::vector<void*>::reverse_iterator it = cudahostfree_array.rbegin(); it != cudahostfree_array.rend();
				++it) {
			cudaErrchk(cudaFreeHost(*it));
			*it = NULL;
		}
		cudahostfree_array.clear();
		SYSDEBUG_LOGGER << "...finished cudaFreeHost(.)" << ENDL;
	}
};

#endif /* memory_manager_GPU_CUH_ */
