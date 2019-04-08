/*
 * SGDLM.cuh
 *
 *  Created on: Dec 2, 2013
 *      Author: lutz
 */

#ifndef SGDLM_CUH_
#define SGDLM_CUH_

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

namespace SGDLM {
template<typename DOUBLE>
class SGDLM {
public:
	// the non-const variablas Q_t, e_t, A_t will be used as temporary variables; their memory has to be allocated already, but the values do not have to be initialized
	// Q_t: m array
	// e_t: m array
	// A_t: m array of pointers to p array
	static void compute_posterior(const DOUBLE* zero, const DOUBLE* plus_one, const DOUBLE* minus_one, size_t m,
			size_t max_p, const unsigned int* p, DOUBLE** m_t, DOUBLE** C_t, DOUBLE* n_t, DOUBLE* s_t,
			const DOUBLE* const y_t, const DOUBLE** const F_t, DOUBLE* Q_t, DOUBLE* e_t, DOUBLE** A_t,
			DOUBLE** Q_t_ptrptr, DOUBLE** e_t_ptrptr, cublasHandle_t CUBLAS, cudaStream_t stream); // register new data

	static void compute_one_step_ahead_prior(size_t m, size_t max_p, const unsigned int* p, DOUBLE** m_t, DOUBLE** C_t,
			DOUBLE* n_t, DOUBLE* s_t, const DOUBLE* beta, const DOUBLE** delta, cudaStream_t stream,
			cublasHandle_t CUBLAS = NULL, const DOUBLE* zero = NULL, const DOUBLE* plus_one = NULL, const DOUBLE** G_t =
					NULL, DOUBLE** C_t_buffer = NULL, DOUBLE** m_t_buffer = NULL); // change the time t posterior parameters to time t+1 prior parameters

	static void VB_posterior(const DOUBLE* zero, const DOUBLE* plus_one, size_t m, size_t max_p, const unsigned int* p,
			const unsigned int* sp_indices, const DOUBLE** m_t, const DOUBLE** C_t, const DOUBLE* n_t,
			const DOUBLE* s_t, size_t n, DOUBLE** lambdas, DOUBLE* randoms, DOUBLE* randoms_pt2,
			DOUBLE** randoms_nrepeat_ptr, DOUBLE** Gammas, size_t Gammas_batch_size, DOUBLE* IS_weights,
			DOUBLE* sum_unnormalized_IS_weights, DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas,
			DOUBLE** thetas_nrepeat_ptr, int* LU_pivots, int* LU_infos, DOUBLE* mean_lambdas, DOUBLE* mean_log_lambdas,
			DOUBLE** mean_m_t, DOUBLE** mean_C_t, DOUBLE** C_t_buffer, int* INV_pivots, int* INV_infos,
			DOUBLE* mean_n_t, DOUBLE* mean_s_t, DOUBLE* mean_Q_t, DOUBLE** Q_t, DOUBLE** Q_t_array_ptr,
			cudaStream_t stream, cublasHandle_t CUBLAS, curandGenerator_t CURAND, bool do_forecast = false,
			const DOUBLE** const x_tp1 = NULL, DOUBLE** y_tp1_nrepeat_ptr = NULL, DOUBLE* data_nus = NULL,
			DOUBLE** nus_nrepeat_ptr = NULL, DOUBLE** Gammas_inv = NULL); // do importance sampling for variational Bayes and include forecasting

	static void forecast(const DOUBLE* zero, const DOUBLE* plus_one, size_t m, size_t max_p, const unsigned int* p,
			const unsigned int* sp_indices, const DOUBLE** m_t, const DOUBLE** C_t, const DOUBLE* n_t,
			const DOUBLE* s_t, size_t n, size_t Gammas_batch_size, const DOUBLE** const x_tp1,
			DOUBLE** y_tp1_nrepeat_ptr, DOUBLE* data_nus, DOUBLE** nus_nrepeat_ptr, DOUBLE** lambdas, DOUBLE* randoms,
			DOUBLE* randoms_pt2, DOUBLE** randoms_nrepeat_ptr, DOUBLE** Gammas, DOUBLE** Gammas_inv, int* INV_pivots,
			int* INV_infos, DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas,
			DOUBLE** thetas_nrepeat_ptr, cudaStream_t stream, cublasHandle_t CUBLAS, curandGenerator_t CURAND); // simulate the t+1 observations from the prior distribution

private:
	static void sample_parameters(const DOUBLE* zero, const DOUBLE* plus_one, size_t m, size_t max_p,
			const unsigned int* p, const DOUBLE** m_t, const DOUBLE** C_t, const DOUBLE* n_t, const DOUBLE* s_t,
			size_t n, DOUBLE** lambdas, DOUBLE* randoms, DOUBLE* randoms_pt2, DOUBLE** randoms_nrepeat_ptr,
			DOUBLE** chol_C_t, DOUBLE** chol_C_t_nrepeat_ptr, DOUBLE** thetas, DOUBLE** thetas_nrepeat_ptr,
			cudaStream_t stream, cublasHandle_t CUBLAS, curandGenerator_t CURAND);
};

}

//#include "SGDLM_impl.cuh"

#endif /* SGDLM_CUH_ */
