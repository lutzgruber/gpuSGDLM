/*
 * utils.cuh
 *
 *  Created on: Dec 7, 2013
 *      Author: lutz
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include "loggerGPU.cuh"
//#include "memory_manager.cuh"

#include <cuda_runtime.h>

/**
 * user must free the out pointer after use!
 */
template<typename T> inline void makePtrArray(T* in, size_t no_matrices, size_t matrix_size, T** out) {
	out = (T**) malloc(no_matrices * sizeof(T*));
	for (size_t i = 0; i < no_matrices; i++) {
		out[i] = (T*) ((char*) in + i * matrix_size * sizeof(T));
	}
}

/**
 * this function frees the host_ptr
 * user must free the dev_ptr after use
 */
template<typename T> inline void cpyPtrArrayToDevice(T** host_ptr, T** dev_ptr, size_t no_matrices) {
	cudaErrchk(cudaMalloc((void** )&dev_ptr, no_matrices * sizeof(T*)));
	cudaErrchk(cudaMemcpy(dev_ptr, host_ptr, no_matrices * sizeof(T*), cudaMemcpyHostToDevice));
	free(host_ptr);
}

#endif /* UTILS_CUH_ */
