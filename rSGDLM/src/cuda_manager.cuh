/*
 * cuda_manager.cuh
 *
 *  Created on: Jan 16, 2014
 *      Author: lutz
 */

#ifndef CUDA_MANAGER_CUH_
#define CUDA_MANAGER_CUH_

#include <cuda_runtime.h>

#include "loggerGPU.cuh"

inline void startCuda(size_t device_id, bool force = false) {
	int current_id = -1;
	cudaErrchk(cudaGetDevice(&current_id));

	if (current_id != device_id || force) {
		cudaErrchk(cudaSetDevice(device_id));
	}
}

#endif /* CUDA_MANAGER_CUH_ */
