/*
 * simPointers.cuh
 *
 *  Created on: Jul 22, 2015
 *      Author: lutz
 */

#ifndef SIMPOINTERS_CUH_
#define SIMPOINTERS_CUH_

namespace SGDLM {
template<typename DOUBLE, class memory_manager> struct simPointers {
	// ALL POINTERS RESIDE ON THE GPU

	// POINTERS FOR SGDLM OBJECTS
	unsigned int* sp_indices;
	unsigned int* p;

	DOUBLE* s_t;
	DOUBLE* n_t;
	DOUBLE* data_m_t;
	DOUBLE** m_t;
	DOUBLE* data_m_t_buffer;
	DOUBLE** m_t_buffer;
	DOUBLE* data_C_t;
	DOUBLE** C_t;
	DOUBLE* data_C_t_buffer;
	DOUBLE** C_t_buffer;

	DOUBLE* alpha;
	DOUBLE* beta;
	DOUBLE* data_delta;
	DOUBLE** delta;

	DOUBLE* data_G_t;
	DOUBLE** G_t;

	DOUBLE* Q_t;
	DOUBLE** Q_t_ptrptr;
	DOUBLE* e_t;
	DOUBLE** e_t_ptrptr;
	DOUBLE* data_A_t;
	DOUBLE** A_t;

	DOUBLE* y_t;
	DOUBLE* data_F_t;
	DOUBLE** F_t;

	DOUBLE* zero;
	DOUBLE* plus_one;
	DOUBLE* minus_one;

	// BEGIN SIMULATION GPU POINTERS
	// VB posterior simulation AND forecasting
	DOUBLE* data_lambdas;
	DOUBLE** lambdas;
	DOUBLE* data_randoms;
	DOUBLE* data_randoms_pt2;
	DOUBLE** randoms_nrepeat_ptr;
	DOUBLE* data_Gammas;
	DOUBLE** Gammas;
	int* LU_pivots;
	int* LU_infos;
	DOUBLE* data_chol_C_t;
	DOUBLE** chol_C_t;
	DOUBLE** chol_C_t_nrepeat_ptr;
	DOUBLE* data_thetas;
	DOUBLE** thetas;
	DOUBLE** thetas_nrepeat_ptr;

	// only VB posterior simulation
	DOUBLE* IS_weights;
	DOUBLE* sum_det_weights;
	DOUBLE* mean_lambdas;
	DOUBLE* mean_log_lambdas;
	DOUBLE* data_mean_m_t;
	DOUBLE* data_mean_C_t;
	DOUBLE** mean_m_t;
	DOUBLE** mean_C_t;
	int* INV_pivots;
	int* INV_infos;
	DOUBLE** lambdas_nrepeat_ptr;
	DOUBLE* mean_n_t;
	DOUBLE* mean_s_t;
	DOUBLE* mean_Q_t;

	// only forecasting
	DOUBLE* data_y;
	DOUBLE** y;
	DOUBLE* data_x_t;
	DOUBLE** x_t;
	DOUBLE* data_nus;
	DOUBLE** nus;
	DOUBLE* data_Gammas_inv;
	DOUBLE** Gammas_inv;

	// number of simulations
	size_t nsim;

	// memory manager
	memory_manager MEM;
	memory_manager MEM_evo;
	memory_manager MEM_sim;

	// Stream for asynchronous command execution
	cudaStream_t stream;

	// cuBlas
	cublasHandle_t CUBLAS;

	// cuRand
	curandGenerator_t CURAND;
};

}

#endif /* SIMPOINTERS_CUH_ */
