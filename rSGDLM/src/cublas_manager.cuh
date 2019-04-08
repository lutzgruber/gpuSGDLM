/*
 * cublas_manager.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: lutz
 */

#ifndef CUBLAS_MANAGER_CUH_
#define CUBLAS_MANAGER_CUH_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "loggerGPU.cuh"
//#include "gpu_tweaks.cuh"

inline bool startCublas(cublasHandle_t* handlePtr) {
	bool success = cublasErrchk(cublasCreate(handlePtr));

	if (success) {
		int cublas_version = 0;
		cublasErrchk(cublasGetVersion(*handlePtr, &cublas_version));
		INFO_LOGGER << "cuBLAS version: " << cublas_version << ENDL;
	}

	return success;
}

inline bool endCublas(cublasHandle_t handle) {
	return cublasErrchk(cublasDestroy(handle));
}

inline void cublasXger(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx,
		const double *y, int incy, double *A, int lda) {
	cublasErrchk(cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

inline void cublasXger(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx,
		const float *y, int incy, float *A, int lda) {
	cublasErrchk(cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
		int k, const double *alpha, /* host or device pointer */
		const double *Aarray[], int lda, const double *Barray[], int ldb, const double *beta, /* host or device pointer */
		double *Carray[], int ldc, int batchCount) {
	cublasErrchk(
			cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
					batchCount));
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
		int k, const float *alpha, /* host or device pointer */
		const float *Aarray[], int lda, const float *Barray[], int ldb, const float *beta, /* host or device pointer */
		float *Carray[], int ldc, int batchCount) {
	cublasErrchk(
			cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
					batchCount));
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, double *A[], int lda, int *P, int *INFO, int batchSize) {
	cublasErrchk(cublasDgetrfBatched(handle, n, A, lda, P, INFO, batchSize));
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, float *A[], int lda, int *P, int *INFO, int batchSize) {
	cublasErrchk(cublasSgetrfBatched(handle, n, A, lda, P, INFO, batchSize));
}

inline void cublasXgetriBatched(cublasHandle_t handle, int n, const double *Aarray[], int lda, int *PivotArray,
		double *Carray[], int ldc, int *infoArray, int batchSize) {
	cublasErrchk(cublasDgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize));
}

inline void cublasXgetriBatched(cublasHandle_t handle, int n, const float *Aarray[], int lda, int *PivotArray,
		float *Carray[], int ldc, int *infoArray, int batchSize) {
	cublasErrchk(cublasSgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize));
}

inline void cublasXasum(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
	cublasErrchk(cublasDasum(handle, n, x, incx, result));
}

inline void cublasXasum(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
	cublasErrchk(cublasSasum(handle, n, x, incx, result));
}

inline void cublasXcopy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
	cublasErrchk(cublasDcopy(handle, n, x, incx, y, incy));
}

inline void cublasXcopy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
	cublasErrchk(cublasScopy(handle, n, x, incx, y, incy));
}

inline void cublasXscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
	cublasErrchk(cublasDscal(handle, n, alpha, x, incx));
}

inline void cublasXscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
	cublasErrchk(cublasSscal(handle, n, alpha, x, incx));
}

inline void cublasXaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y,
		int incy) {
	cublasErrchk(cublasDaxpy(handle, n, alpha, x, incx, y, incy));
}

inline void cublasXaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y,
		int incy) {
	cublasErrchk(cublasSaxpy(handle, n, alpha, x, incx, y, incy));
}

#endif /* CUBLAS_MANAGER_CUH_ */
