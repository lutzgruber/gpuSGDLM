/*
 * loggerGPU.cuh
 *
 *  Created on: Jul 27, 2015
 *      Author: lutz
 */

#ifndef LOGGERGPU_CUH_
#define LOGGERGPU_CUH_

#include "logger.hpp"


#ifdef USE_MATLAB
#ifdef MATLAB_GPU
#include "gpu/mxGPUArray.h"
#endif
#endif


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "cublasGetErrorString.cuh"
#include "curandGetErrorString.cuh"


inline bool cublasAssert(cublasStatus_t code, const char* file, int line) {
	if (code != CUBLAS_STATUS_SUCCESS) {
		int no_gpu = -1;
		cudaGetDevice(&no_gpu);
		WARNING_LOGGER << "cublasAssert (GPU " << no_gpu << "): " << cublasGetErrorString(code) << " in "
				<< std::string(file) << ", line " << line << "." << ENDL;
	}

	return code == CUBLAS_STATUS_SUCCESS;
}

inline bool curandAssert(curandStatus_t code, const char* file, int line) {
	if (code != CURAND_STATUS_SUCCESS) {
		int no_gpu = -1;
		cudaGetDevice(&no_gpu);
		WARNING_LOGGER << "curandAssert (GPU " << no_gpu << "): " << curandGetErrorString(code) << " in "
				<< std::string(file) << ", line " << line << "." << ENDL;
	}

	return code == CURAND_STATUS_SUCCESS;
}

inline bool cudaAssert(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		int no_gpu = -1;
		cudaGetDevice(&no_gpu);
		WARNING_LOGGER << "cudaAssert (GPU " << no_gpu << "): " << cudaGetErrorString(code) << " in "
				<< std::string(file) << ", line " << line << "." << ENDL;
	}

	return code == cudaSuccess;
}

#ifdef USE_MATLAB
#ifdef MATLAB_GPU
inline bool mexGPUAssert(int code, const char* file, int line) {
	if (code != MX_GPU_SUCCESS) {
		WARNING_LOGGER << "mexGPUAssert: could not initialize the Mathworks GPU API in "
				<< std::string(file) << ", line " << line << "." << ENDL;
	}

	return code == MX_GPU_SUCCESS;
}
#define mxGPUErrchk(ans) { mexGPUAssert((ans), __FILE__, __LINE__); }
#endif
#endif

#define cublasErrchk(ans) cublasAssert((ans), __FILE__, __LINE__);
#define curandErrchk(ans) curandAssert((ans), __FILE__, __LINE__);
#define cudaErrchk(ans) cudaAssert((ans), __FILE__, __LINE__);



#endif /* LOGGERGPU_CUH_ */
