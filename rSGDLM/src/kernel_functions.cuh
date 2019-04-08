/*
 * kernel_functions.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: lutz
 */

#ifndef GPU_TWEAKS_CUH_
#define GPU_TWEAKS_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>

//#define THREADS_PER_BLOCK 512
//#define OPTIM_PRECISION 1e-6
//#define OPTIM_MAX_ITER 50

enum SCALE_TRANSFORMATION {
	SCALE_TRANSFORMATION_NONE, SCALE_TRANSFORMATION_LOG, SCALE_TRANSFORMATION_SQRT, SCALE_TRANSFORMATION_INV
};

template<typename NUM> __global__ void sumVector(size_t vector_length, const NUM* vector, NUM* sum_ptr) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i == 0) {
		NUM sum = 0;

		for (size_t k = 0; k < vector_length; k++) {
			sum += vector[k];
		}

		sum_ptr[0] = sum;
	}
}

template<typename NUM> __global__ void assignScalar(size_t vector_length, NUM* device_scalar_pointer, NUM value) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < vector_length) {
		device_scalar_pointer[i] = value;
	}
}

/**
 * v1, v2 are arrays of size length
 * alpha is a scalar
 *
 * This function computes:
 *
 * for j=1:length
 *   v2(j) = alpha * v1(j) * v2(j)
 * end
 *
 */
template<typename NUM> __global__ void vvScale(size_t length, NUM alpha, const NUM* v1, NUM* v2) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length) {
		v2[i] = alpha * v1[i] * v2[i];
	}
}

template<typename NUM> __global__ void makeVB_MVN(size_t batchCount, size_t m, size_t max_p, const NUM** m_t,
		const NUM** lambdas, const NUM* s_t, NUM** MVNs) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchCount
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // m
	size_t k = blockIdx.z * blockDim.z + threadIdx.z; // max_p

	if (i < batchCount && j < m && k < max_p) {
		MVNs[i][j * max_p + k] = m_t[j][k] + MVNs[i][j * max_p + k] / sqrt(s_t[j] * lambdas[i][j]);
	}
}

template<typename NUM> __global__ void batchedCutOffVector(size_t batchCount, size_t vector_length,
		const unsigned int* cutoff_points, NUM** V) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchCount
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // vector_length

	if (i < batchCount && j < vector_length && j >= cutoff_points[i]) {
		V[i][j] = 0;
	}
}

template<typename NUM> __global__ void batchedCutOffMatrix(size_t batchCount, size_t nrows, size_t ncols,
		const unsigned int* cutoff_rows, const unsigned int* cutoff_cols, NUM** M) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchCount
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // nrows * ncols

	if (i < batchCount && j < nrows * ncols) {
		if (j % nrows >= cutoff_rows[i] && j >= nrows * cutoff_cols[i]) {
			M[i][j] = 0;
		}
	}
}

template<typename NUM> __global__ void batchedSetOutsideSquareMatrixDiagonal(size_t batchCount, size_t matrix_dim,
		const unsigned int* cutoff_dim, NUM** M, NUM value) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchCount
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // matrix_dim

	if (i < batchCount && j < matrix_dim) {
		if (j >= cutoff_dim[i]) {
			M[i][j * matrix_dim + j] = value;
		}
	}
}

/*
 * V is an array of batchCount pointers to p-length arrays
 * c is a batchCount-length array
 *
 * The output is written to V.
 *
 * This function computes:
 *
 * for j=1:batchCount
 * 	 V(j,:) = V(j,:) * T(c(j))
 * end
 *
 */
template<typename NUM> __global__ void batchedScale(size_t batchCount, size_t p, NUM** V, const NUM* c,
		SCALE_TRANSFORMATION T = SCALE_TRANSFORMATION_NONE) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchCount
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // p

	if (i < batchCount && j < p) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i][j] = V[i][j] * sqrt(c[i]);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i][j] = V[i][j] * log(c[i]);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i][j] = V[i][j] / c[i];
			break;
		default:
			V[i][j] = V[i][j] * c[i];
		}
	}
}

template<typename NUM> __global__ void scale(size_t vector_length, NUM* V, const NUM* c, SCALE_TRANSFORMATION T =
		SCALE_TRANSFORMATION_NONE) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // vector_length

	if (i < vector_length) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i] = V[i] * sqrt(*c);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i] = V[i] * log(*c);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i] = V[i] / (*c);
			break;
		default:
			V[i] = V[i] * (*c);
		}
	}
}

/*
 * V is an array of batchCount pointers to p-length arrays which contain matrix data in column-major form
 * ... change only the first change_cols columns and change_rows rows
 * c is a batchCount-length array
 *
 * The output is written to V.
 *
 * This function computes:
 *
 * for j=1:batchCount
 * 	 V(j,:) = V(j,:) / T(c(j))
 * end
 *
 */
template<typename NUM> __global__ void batchedScale(size_t batchCount, size_t nrows, const unsigned int* change_cols,
		const unsigned int* change_rows, NUM** V, const NUM* c, SCALE_TRANSFORMATION T = SCALE_TRANSFORMATION_NONE) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < batchCount && j % nrows < change_rows[i] && j < nrows * change_cols[i]) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i][j] = V[i][j] * sqrt(c[i]);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i][j] = V[i][j] * log(c[i]);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i][j] = V[i][j] / c[i];
			break;
		default:
			V[i][j] = V[i][j] * c[i];
		}
	}
}

template<typename NUM> __global__ void batchedMatrixScale(size_t batchCount, size_t nrows,
		const unsigned int* change_cols, const unsigned int* change_rows, NUM** V, const NUM** c,
		SCALE_TRANSFORMATION T = SCALE_TRANSFORMATION_NONE) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	size_t row = j % nrows;
	size_t col = j / nrows;

	if (i < batchCount && row < change_rows[i] && col < change_cols[i]) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i][j] *= sqrt(c[i][j]);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i][j] *= log(c[i][j]);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i][j] /= c[i][j];
			break;
		default:
			V[i][j] *= c[i][j];
		}
	}
}

template<typename NUM> __global__ void batchedMatrixDiagScale(size_t batchCount, size_t nrows,
		const unsigned int* change_cols, const unsigned int* change_rows, NUM** V, const NUM* c,
		SCALE_TRANSFORMATION T = SCALE_TRANSFORMATION_NONE) {
	extern __shared__ NUM scale_entries[];

	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < batchCount && threadIdx.y < nrows) {
		scale_entries[threadIdx.x * nrows + threadIdx.y] = sqrt(c[i * nrows + threadIdx.y]);
	}

	__syncthreads();

	size_t row = j % nrows;
	size_t col = j / nrows;

	if (i < batchCount && row < change_rows[i] && col < change_cols[i]) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i][j] *= sqrt(scale_entries[threadIdx.x * nrows + row] * scale_entries[threadIdx.x * nrows + col]);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i][j] *= log(scale_entries[threadIdx.x * nrows + row] * scale_entries[threadIdx.x * nrows + col]);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i][j] /= (scale_entries[threadIdx.x * nrows + row] * scale_entries[threadIdx.x * nrows + col]);
			break;
		default:
			V[i][j] *= scale_entries[threadIdx.x * nrows + row] * scale_entries[threadIdx.x * nrows + col];
		}
	}
}

/*
 * V is an array of batchCount pointers to p-length arrays
 * c is a pointer to a scalar
 *
 * The output is written to V.
 *
 * This function computes:
 *
 * for j=1:batchCount
 * 	 V(j,:) = V(j,:) / T(c)
 * end
 *
 */
template<typename NUM> __global__ void batchedScalarScale(size_t batchCount, size_t p, NUM** V, const NUM* c,
		SCALE_TRANSFORMATION T = SCALE_TRANSFORMATION_NONE) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < batchCount && j < p) {
		switch (T) {
		case SCALE_TRANSFORMATION_SQRT:
			V[i][j] = V[i][j] * sqrt(*c);
			break;
		case SCALE_TRANSFORMATION_LOG:
			V[i][j] = V[i][j] * log(*c);
			break;
		case SCALE_TRANSFORMATION_INV:
			V[i][j] = V[i][j] / *c;
			break;
		default:
			V[i][j] = V[i][j] * *c;
		}
	}
}

/**
 * m_t, A_t are arrays of batchCount pointers to p-length arrays
 * e_t is a batchCount-length array
 *
 * The output is written to m_t.
 *
 * This function computes:
 *
 * for j=1:batchCount
 *   m_t(j,:) = m_t(j,:) + e_t(j) * A_t(j,:)
 * end
 */
template<typename NUM> __global__ void batchedComputeMt(size_t batchCount, size_t p, NUM** m_t, const NUM** A_t,
		const NUM* e_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < batchCount && j < p) {
		m_t[i][j] = m_t[i][j] + e_t[i] * A_t[i][j];
	}
}

template<typename NUM> __global__ void batchedScaleCt(size_t batchCount, size_t no_elements, NUM** C_t, const NUM* n_t,
		const NUM* e_t, const NUM* Q_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // m
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // no_elements

	if (i < batchCount && j < no_elements) {
		C_t[i][j] *= ((n_t[i] - 1) + e_t[i] * e_t[i] / Q_t[i]) / n_t[i];
	}
}

/*
 * V is a vector of length length
 * c is a constant
 *
 * The output is written to V.
 *
 * This function computes:
 *
 * for j=1:length
 *   V(j) = V(j) + c
 * end
 *
 */
template<typename NUM> __global__ void addScalar(size_t length, NUM* V, NUM c) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length) {
		V[i] = V[i] + c;
	}
}

/*
 * V1, V2 are a vector of length length
 *
 * The output is written to V1.
 *
 * This function computes:
 *
 * for j=1:length
 *   V1(j) = V1(j) * V2(j)
 * end
 *
 */
template<typename NUM> __global__ void multiply(size_t length, NUM* V1, const NUM* V2) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length) {
		V1[i] = V1[i] * V2[i];
	}
}

/*
 * s_t, n_t, e_t, Q_t are vectors of length length
 *
 * This output is written to s_t.
 *
 * This function computes:
 *
 * for j=1:length
 *   s_t(j) = s_t(j) * (n_t(i) - 1 + e_t(i) * e_t(i) / Q_t(i)) / n_t(i)
 * end
 */
template<typename NUM> __global__ void computeSt(size_t length, NUM* s_t, const NUM* n_t, const NUM* e_t,
		const NUM* Q_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length) {
		s_t[i] = s_t[i] * (n_t[i] - ((NUM) 1) + e_t[i] * e_t[i] / Q_t[i]) / n_t[i];
	}
}

/**
 * fill_indices is a matrix of size m x no_blocks; it should contain the number of gamma r.v.'s that are already calculated per block
 * uniforms is a vector of n_loops * n * m U(0,1) random variables
 * normals is a vector of n_loops * n * m N(0,1) random variables
 * alpha, beta are vectors of size m; they are the parameters of the gamma distribution which we want to sample
 * gammas is a matrix of size m x n; the gamma random variables will be stored there
 */
template<typename NUM> __global__ void make_gamma(size_t m, size_t n, size_t n_loops, unsigned int** fill_indices,
		const NUM* uniforms, const NUM* normals, const NUM* alpha, const NUM* beta, NUM** gammas) {
	extern __shared__ unsigned int counter[];
	size_t base_i = blockIdx.x * blockDim.x; // note that we did not include the thread count---this will be replaced by counter!
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < m && alpha[j] >= 1) {
		NUM beta_ = beta[j];

		NUM d = alpha[j] - 1.0 / 3.0;
		NUM c = 1 / sqrt(9 * d);

		if (threadIdx.x == 0) {
			counter[j] = fill_indices[j][blockIdx.x];
		}

		// initialize to zero
		if (base_i + threadIdx.x < n) {
			gammas[base_i + threadIdx.x][j] = 0;
		}
		__syncthreads();

		if (base_i + threadIdx.x < n) {
			for (size_t k = 0; k < n_loops; k++) {
				size_t rand_index = j * n_loops * n + k * n + base_i + threadIdx.x;

				NUM x = normals[rand_index];

				NUM v = (1 + c * x) * (1 + c * x) * (1 + c * x);

				if (v <= 0) {
					continue;
				}

				NUM x_sq = x * x;

				NUM U = uniforms[rand_index];

				if (U < 1 - 0.331 * x_sq * x_sq || log(U) < 0.5 * x_sq + d * (1 - v + log(v))) {
					size_t i = atomicAdd(&counter[j], 1);

					if (i < blockDim.x && base_i + i < n) {
						gammas[base_i + i][j] = d * v / beta_;
					}
				}
			}
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			fill_indices[j][blockIdx.x] = counter[j];
		}
	}
}

/**
 * fill_indices is a matrix of size m x no_blocks; it should contain the number of gamma r.v.'s that are already calculated per block
 * uniforms is a vector of n_loops * n * m U(0,1) random variables
 * normals is a vector of n_loops * n * m N(0,1) random variables
 * alpha, beta are vectors of size m; they are the parameters of the gamma distribution which we want to sample
 * gammas is a matrix of size m x n; the gamma random variables will be stored there
 */
template<typename NUM> __global__ void make_gamma2(size_t m, size_t n, size_t n_loops, unsigned int** fill_indices,
		const NUM* uniforms, const NUM* normals, const NUM* n_t, const NUM* s_t, NUM** gammas) {
	extern __shared__ unsigned int counter[];
	size_t base_i = blockIdx.x * blockDim.x; // note that we did not include the thread count---this will be replaced by counter!
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // m

	if (j < m && n_t[j] >= 2 /* this is equivalent to testing if the alpha parameter is >= 1 */) {
		NUM alpha_ = 0.5 * n_t[j];
		NUM beta_ = 0.5 * n_t[j] * s_t[j];

		NUM d = alpha_ - 1.0 / 3.0;
		NUM c = 1 / sqrt(9 * d);

		if (threadIdx.x == 0) {
			if (fill_indices != NULL) {
				counter[j] = fill_indices[j][blockIdx.x];
			} else {
				counter[j] = 0;
			}
		}

		// initialize to zero
		if (base_i + threadIdx.x < n) {
			gammas[base_i + threadIdx.x][j] = 0;
		}

		__syncthreads();

		if (base_i + threadIdx.x < n) {
			for (size_t k = 0; k < n_loops; k++) {
				size_t rand_index = j * n_loops * n + k * n + base_i + threadIdx.x;

				NUM x = normals[rand_index];

				NUM v = (1 + c * x) * (1 + c * x) * (1 + c * x);

				if (v <= 0) {
					continue;
				}

				NUM x_sq = x * x;

				NUM U = uniforms[rand_index];

				if (U < 1 - 0.331 * x_sq * x_sq || log(U) < 0.5 * x_sq + d * (1 - v + log(v))) {
					size_t i = atomicAdd(&counter[j], 1);

					if (i < blockDim.x && base_i + i < n) {
						gammas[base_i + i][j] = d * v / beta_;
					}
				}
			}
		}

		__syncthreads();

		if (threadIdx.x == 0 && fill_indices != NULL) {
			fill_indices[j][blockIdx.x] = counter[j];
		}
	}
}

template<typename NUM> __global__ void initVector(size_t vector_length, NUM* vector) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // vector_length

	if (i < vector_length) {
		vector[i] = 0;
	}
}

template<typename NUM> __global__ void batchedInitVector(size_t batchSize, size_t vector_length, NUM** vectors) {
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // y dimension = batches
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // x dimension = length of the vectors

	if (j < batchSize && i < vector_length) {
		vectors[j][i] = 0;
	}
}

template<typename NUM> __global__ void fillGammas(size_t batchSize, size_t batch_offset, size_t m, size_t max_p,
		const unsigned int* sp_indices_linear, NUM** Gammas, const NUM** thetas) {
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // y dimension = batches
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // x dimension = m*max_p entries of the Gammas/gammas

	if (j < batchSize && i < m * max_p) {
		if (sp_indices_linear[i] < m * m) {
			// Gammas is organized by MC batch -> each entry is a m x m matrix
			// thetas is organized by m (dimensions) -> each entry is a concatenation of max_p -length vectors
			Gammas[j][sp_indices_linear[i]] = -thetas[batch_offset + j][i];
		}

		// fill diagonal entries with 1
		if (i < m) {
			Gammas[j][i * m + i] = 1;
		}
	}
}

template<typename NUM> __global__ void cholesky(size_t m, size_t ld, size_t max_p, const unsigned int* p,
		const NUM** matrices, NUM** chol_matrices) {
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // y dimension = batches

	if (j < m) {
		unsigned int p_ = p[j];

		/* copy matrices into chol_matrices */
		for (size_t n = 0; n < p_; n++) {
			for (size_t k = 0; k < n; k++) {
				chol_matrices[j][k * ld + n] = 0;
			}
			for (size_t k = n; k < p_; k++) {
				chol_matrices[j][k * ld + n] = matrices[j][k * ld + n];
			}
			for (size_t k = p_; k < max_p; k++) {
				chol_matrices[j][k * ld + n] = 0;
			}
		}
		for (size_t n = p_; n < max_p; n++) {
			for (size_t k = 0; k < max_p; k++) {
				chol_matrices[j][k * ld + n] = 0;
			}
		}

		/* sweep down the matrix... */
		for (size_t n = 0; n < p_; n++) {
			/* do the diagonal element... */
			NUM diag_n = sqrt(chol_matrices[j][n * ld + n]);
			chol_matrices[j][n * ld + n] = diag_n; // matrix[n][n] = sqrt(matrix[n][n]);

			NUM a = 1.0 / diag_n; // a = 1. / sqrt(matrix[n][n]);

			for (size_t k = n + 1; k < p_; k++) {
				/* do the top strip...*/
				chol_matrices[j][k * ld + n] *= a; // matrix[n][k] = matrix[n][k] * a;

				/* update the diagonal... */
				chol_matrices[j][k * ld + k] -= chol_matrices[j][k * ld + n] * chol_matrices[j][k * ld + n]; // matrix[k][k] = matrix[k][k] - matrix[n][k] * matrix[n][k];
			}

			/* update the rest... */
			for (size_t k = n + 1; k < p_; k++) {
				for (size_t l = k + 1; l < p_; l++) {
					chol_matrices[j][l * ld + k] -= chol_matrices[j][k * ld + n] * chol_matrices[j][l * ld + n]; // matrix[k][l] = matrix[k][l] - matrix[n][k] * matrix[n][l];
				}
			}

		}
	}
}

template<typename NUM> __global__ void detFromLU(size_t batchSize, size_t matrix_dim, const NUM** matrices,
		NUM* determinants, size_t IS_index) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchSize

	if (i < batchSize) {
		NUM log_determinant = 0;

		for (size_t x = 0; x < matrix_dim; x++) {
			log_determinant += log(fabs(matrices[i][x * matrix_dim + x]));
		}

		// exponentiate to obtain the actual determinant
		log_determinant = exp(log_determinant);

		if (isnan(log_determinant) || isinf(log_determinant)) {
			determinants[IS_index + i] = 0;
		} else {
			determinants[IS_index + i] = log_determinant;
		}
	}
}

template<typename NUM> __global__ void meanScalarAndLog(size_t batchSize, size_t m, const NUM* batch_weights,
		const NUM** scalars, NUM* mean_scalars, NUM* mean_log_scalars) {
	size_t j = blockIdx.x * blockDim.x + threadIdx.x; // m

	if (j < m) {
		NUM mean_scalars_ = 0;
		NUM mean_log_scalars_ = 0;

		for (size_t k = 0; k < batchSize; k++) {
			mean_scalars_ += batch_weights[k] * scalars[k][j];
			mean_log_scalars_ += batch_weights[k] * log(scalars[k][j]);
		}

		mean_scalars[j] = mean_scalars_;
		mean_log_scalars[j] = mean_log_scalars_;
	}
}

template<typename NUM> __global__ void compute_mean_m_t(size_t batchSize, size_t m, size_t max_p, const NUM* IS_weights,
		const NUM** lambdas, const NUM* mean_lambdas, const NUM** thetas, NUM** mean_m_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // max_p
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // m

	if (i < max_p && j < m) {
		NUM mean_m_t_ji = 0;

		for (size_t k = 0; k < batchSize; k++) {
			mean_m_t_ji += IS_weights[k] * lambdas[k][j] * thetas[k][j * max_p + i];
		}

		mean_m_t_ji /= mean_lambdas[j];

		mean_m_t[j][i] = mean_m_t_ji;
	}
}

// replaces thetas by sqrt(lambdas) * (thetas - mean_m_t)
template<typename NUM> __global__ void compute_VB_vector1(size_t batchSize, size_t m, size_t max_p, const NUM** lambdas,
		const NUM** mean_m_t, NUM** thetas) {
	size_t i = blockIdx.z * blockDim.z + threadIdx.z; // max_p
	size_t j = blockIdx.x * blockDim.x + threadIdx.x; // m
	size_t k = blockIdx.y * blockDim.y + threadIdx.y; // batchSize

	if (i < max_p && j < m && k < batchSize) {
		thetas[k][j * max_p + i] = sqrt(lambdas[k][j]) * (thetas[k][j * max_p + i] - mean_m_t[j][i]);
	}
}

/*
 * calculate the digamma function
 */
template<typename NUM> __device__    inline NUM psi(NUM x) {
	return log(x) - 1 / (2 * x) - 1 / (12 * x * x) + 1 / (120 * x * x * x * x) - 1 / (252 * x * x * x * x * x * x)
			+ 1 / (240 * x * x * x * x * x * x * x * x) - 5 / (660 * x * x * x * x * x * x * x * x * x * x)
			+ 691 / (32760 * x * x * x * x * x * x * x * x * x * x * x * x)
			- 1 / (12 * x * x * x * x * x * x * x * x * x * x * x * x * x * x);
}

/*
 * calculate the trigamma function
 */
template<typename NUM> __device__    inline NUM psi1(NUM x) {
	return 1 / x + 1 / (2 * x * x) + 1 / (6 * x * x * x) - 1 / (30 * x * x * x * x * x)
			+ 1 / (42 * x * x * x * x * x * x * x) - 1 / (30 * x * x * x * x * x * x * x * x * x);
}

template<typename NUM> __global__ void compute_mean_n_t(size_t m, const unsigned int* p, const NUM* mean_Q_t,
		const NUM* mean_lambda, const NUM* mean_log_lambda, const NUM* n_t, NUM* mean_n_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // m

	/*if (i < m) {
	 mean_n_t[i] = n_t[i];
	 }
	 return;*/

	if (i < m) {
		size_t iter = 0;

		NUM mean_n_t_ = n_t[i];
		NUM mean_Q_t_ = mean_Q_t[i];
		NUM mean_lambda_ = mean_lambda[i];
		NUM mean_log_lambda_ = mean_log_lambda[i];
		unsigned int p_ = p[i];

		NUM fun, dfun, step = 0;

		do {
			mean_n_t_ -= step;

			fun = log(mean_n_t_ + p_ - mean_Q_t_) - psi(mean_n_t_ / 2) - (p_ - mean_Q_t_) / mean_n_t_
					- log(2 * mean_lambda_) + mean_log_lambda_;
			dfun = 1 / (mean_n_t_ + p_ - mean_Q_t_) - .5 * psi1(mean_n_t_ / 2)
					+ (p_ - mean_Q_t_) / (mean_n_t_ * mean_n_t_);
			step = fun / dfun;

			if (step > mean_n_t_) {
				step = .5 * mean_n_t_;
			}
		} while (abs(fun) > OPTIM_PRECISION && iter++ < OPTIM_MAX_ITER);

		if (iter >= OPTIM_MAX_ITER) {
			mean_n_t[i] = n_t[i];
		} else {
			mean_n_t[i] = mean_n_t_;
		}
	}
}

template<typename NUM> __global__ void compute_mean_s_t(size_t m, const unsigned int* p, const NUM* mean_n_t,
		const NUM* mean_Q_t, const NUM* mean_lambda, const NUM* s_t, NUM* mean_s_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // m

	if (i < m) {
		mean_s_t[i] = (mean_n_t[i] + p[i] - mean_Q_t[i]) / (mean_n_t[i] * mean_lambda[i]);

		if (mean_s_t[i] < 0) {
			mean_s_t[i] = s_t[i];
		}
	}
}

template<typename NUM> __global__ void compute_mean_Q_t(size_t batchSize, size_t m, const NUM* IS_weights,
		const NUM** Q_t, NUM* mean_Q_t) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // m

	if (i < m) {
		mean_Q_t[i] = 0;

		for (size_t k = 0; k < batchSize; k++) {
			mean_Q_t[i] += IS_weights[k] * Q_t[k][i];
		}
	}
}

template<typename NUM> __global__ void meanBatchedVecVecT(size_t batchSize, size_t m, size_t max_p,
		const unsigned int* p, const NUM* IS_weights, const NUM** thetas, NUM** output_matrices) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x; // max_p
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // m
	size_t y = blockIdx.z * blockDim.z + threadIdx.z; // max_p

	if (j < m) {
		if (x < p[j] && y < p[j]) {
			if (x <= y) {
				output_matrices[j][y * max_p + x] = 0;

				for (size_t k = 0; k < batchSize; k++) {
					output_matrices[j][y * max_p + x] += IS_weights[k] * thetas[k][j * max_p + x]
							* thetas[k][j * max_p + y];
				}

				if (x != y) {
					output_matrices[j][x * max_p + y] = output_matrices[j][y * max_p + x];
				}
			}
		} else if (x < max_p && y < max_p) {
			if (x == y) {
				output_matrices[j][y * max_p + x] = 1;
			} else {
				output_matrices[j][y * max_p + x] = 0;
			}
		}
	}
}

template<typename NUM> __global__ void copy(size_t vector_length, const NUM* in, NUM* out) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // vector_length

	if (i < vector_length) {
		out[i] = in[i];
	}
}

template<typename NUM> __global__ void batchedCopy(size_t batchSize, size_t vector_length, const NUM** in, NUM** out) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // vector_length
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // batchSize

	if (j < batchSize && i < vector_length) {
		out[j][i] = in[j][i];
	}
}

template<typename NUM> __global__ void batchedComputeNuPlusMu(size_t batchSize, size_t m, size_t max_p,
		const unsigned int* sp_indices, const NUM** x_t, const NUM** thetas, const NUM** lambdas, NUM** nu) {
	// we defined sp_indices < m*m for simultaneous parents -> if sp_indices >= m*m, the corresponding entry is a "real" parent

	size_t i = blockIdx.x * blockDim.x + threadIdx.x; // batchSize
	size_t j = blockIdx.y * blockDim.y + threadIdx.y; // m

	if (i < batchSize && j < m) {
		NUM nu_ij = nu[i][j];
		nu_ij = nu_ij / sqrt(lambdas[i][j]);

		for (size_t k = 0; k < max_p; k++) {
			if (sp_indices[j * max_p + k] >= m * m) {
				nu_ij += x_t[j][k] * thetas[i][j * max_p + k];
			}
		}

		nu[i][j] = nu_ij;
	}
}

#endif /* GPU_TWEAKS_CUH_ */
