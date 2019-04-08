/*
 * curand_manager.cuh
 *
 *  Created on: Dec 11, 2013
 *      Author: lutz
 */

#ifndef CURAND_MANAGER_CUH_
#define CURAND_MANAGER_CUH_

#include <cuda_runtime.h>
#include <curand.h>

//#ifdef MATLAB
//#include "mx_class_id.hpp"
//#include "memory_manager_MATLAB.cuh"
//#ifdef MATLAB_GPU
//#include "gpu/mxGPUArray.h"
//#endif
//#endif

#include "loggerGPU.cuh"
#include "memory_manager_GPU.cuh"
#include "kernel_functions.cuh"

//#define GAMMA_LOOPS 2

inline bool startCurand(curandGenerator_t* genPtr) {
	return curandErrchk(curandCreateGenerator(genPtr, CURAND_RNG_PSEUDO_DEFAULT));
}

inline bool endCurand(curandGenerator_t gen) {
	return curandErrchk(curandDestroyGenerator(gen));
}

inline void curandGenerateUniformX(curandGenerator_t generator, double* outputPtr, size_t n) {
	curandErrchk(curandGenerateUniformDouble(generator, outputPtr, n));
}

inline void curandGenerateUniformX(curandGenerator_t generator, float* outputPtr, size_t n) {
	curandErrchk(curandGenerateUniform(generator, outputPtr, n));
}

inline void curandGenerateNormalX(curandGenerator_t generator, double* outputPtr, size_t n, double mean,
		double stddev) {
	curandErrchk(curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev));
}

inline void curandGenerateNormalX(curandGenerator_t generator, float* outputPtr, size_t n, float mean,
		float stddev) {
	curandErrchk(curandGenerateNormal(generator, outputPtr, n, mean, stddev));
}

template<typename DOUBLE> inline void sampleGamma(curandGenerator_t gen, memory_manager_GPU& MEM, size_t m, size_t n, const DOUBLE* alphas,
		const DOUBLE* betas, DOUBLE** gammas) {
	size_t n_loops = GAMMA_LOOPS;

	size_t n_input_random_numbers = n * n_loops * m;

//#ifdef MATLAB_GPU
//	mxGPUArray* mx_uniforms = MEM._mxGPUCreateGPUArray(1, &n_input_random_numbers, mx_class_id<DOUBLE>::id, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
//	DOUBLE* uniforms = (DOUBLE*) mxGPUGetData(mx_uniforms);
//#else
	DOUBLE* uniforms = NULL;
	MEM._cudaMalloc(uniforms, n_input_random_numbers * sizeof(DOUBLE));
//#endif
	curandGenerateUniformX(gen, uniforms, n_input_random_numbers);

//#ifdef MATLAB_GPU
//	mxGPUArray* mx_normals = MEM._mxGPUCreateGPUArray(1, &n_input_random_numbers, mx_class_id<DOUBLE>::id, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
//	DOUBLE* normals = (DOUBLE*) mxGPUGetData(mx_normals);
//#else
	DOUBLE* normals = NULL;
	MEM._cudaMalloc(normals, n_input_random_numbers * sizeof(DOUBLE));
//#endif
	curandGenerateNormalX(gen, normals, n_input_random_numbers, 0, 1);

	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
	dim3 numBlocks((n + THREADS_PER_BLOCK - 1) / threadsPerBlock.x, m);

//#ifdef MATLAB_GPU
//	size_t dim_fill[] = { m, numBlocks.x };
//	mxGPUArray* mx_fill_indices = MEM._mxGPUCreateGPUArray(2, dim_fill, mxUINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
//	unsigned int* data_fill_indices = (unsigned int*) mxGPUGetData(mx_fill_indices);
//#else
	unsigned int* data_fill_indices = NULL;
	MEM._cudaMalloc(data_fill_indices, m * numBlocks.x * sizeof(unsigned int));
//#endif
	unsigned int** fill_indices = NULL;
	MEM.cpyToDeviceAsPtrArray(data_fill_indices, m, numBlocks.x, fill_indices);

	//cudaErrchk(cudaDeviceSynchronize());

	make_gamma<<<numBlocks, threadsPerBlock, m * numBlocks.x * sizeof(unsigned int)>>>(m, n, n_loops, fill_indices,
			uniforms, normals, alphas, betas, gammas);
	cudaErrchk(cudaGetLastError());
}

/**
 * memory_uniforms and memory_normals must be of size n * GAMMA_LOOPS * m
 */
template<typename DOUBLE> inline void sampleGamma2(curandGenerator_t gen, size_t m, size_t n, const DOUBLE* n_t,
		const DOUBLE* s_t, DOUBLE* memory_uniforms, DOUBLE* memory_normals, DOUBLE** gammas, cudaStream_t stream = 0) {
	size_t n_loops = GAMMA_LOOPS;

	size_t n_input_random_numbers = n * n_loops * m;

	curandGenerateUniformX(gen, memory_uniforms, n_input_random_numbers);
	curandGenerateNormalX(gen, memory_normals, n_input_random_numbers, 0, 1);

	dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
	dim3 numBlocks((n + THREADS_PER_BLOCK - 1) / threadsPerBlock.x, m);

	/*size_t dim_fill[] = { m, numBlocks.x };
	mxGPUArray* mx_fill_indices = MEM._mxGPUCreateGPUArray(2, dim_fill, mxUINT32_CLASS, mxREAL,
			MX_GPU_INITIALIZE_VALUES);
	unsigned int* data_fill_indices = (unsigned int*) mxGPUGetData(mx_fill_indices);
	unsigned int** fill_indices = NULL;
	MEM.cpyToDeviceAsPtrArray(data_fill_indices, m, numBlocks.x, fill_indices);*/
	unsigned int** fill_indices = NULL;

	//cudaErrchk(cudaDeviceSynchronize());

	make_gamma2<<<numBlocks, threadsPerBlock, m * numBlocks.x * sizeof(unsigned int), stream>>>(m, n, n_loops, fill_indices,
			memory_uniforms, memory_normals, n_t, s_t, gammas);
	cudaErrchk(cudaGetLastError());
}

#endif /* CURAND_MANAGER_CUH_ */
