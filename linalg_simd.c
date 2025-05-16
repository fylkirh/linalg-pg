#include "linalg.h"
#include <immintrin.h>
#include <assert.h>
#include <omp.h>

#define TILE_SIZE 64



static inline float horizontal_add(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void scalarMulVecSimd(const float scalar, const Matrix2D *vec, Matrix2D *result) {
    assert(vec->rows == 1);
    if (vec->cols == 0) return;
    const size_t simd_width = 8;
    size_t i;
    
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (i = 0; i + simd_width <= vec->rows * vec->cols; i += simd_width) {
        __m256 vec_data = _mm256_loadu_ps(&vec->data[i]);
        __m256 mul_result = _mm256_mul_ps(vec_data, scalar_vec);
        _mm256_storeu_ps(&result->data[i], mul_result);
    }
    
    for (; i < vec->rows * vec->cols; ++i) {
        result->data[i] = scalar * vec->data[i];
    }
}

float dotSimd(const Matrix2D *vec1, const Matrix2D *vec2) {
    assert(vec1->cols == vec2->cols);
    assert(vec1->rows == 1 && vec2->rows == 1);
    const size_t simd_width = 8;
    size_t i;
    float res = 0.0f;
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    
    for (i = 0; i + 2 * simd_width <= vec1->cols; i += 2 * simd_width) {
        __m256 vecData11 = _mm256_loadu_ps(&vec1->data[i]);
        __m256 vecData21= _mm256_loadu_ps(&vec2->data[i]);
        accumulator1 = _mm256_fmadd_ps(vecData11, vecData21, accumulator1);

        __m256 vecData12 = _mm256_loadu_ps(&vec1->data[i + simd_width]);
        __m256 vecData22 = _mm256_loadu_ps(&vec2->data[i + simd_width]);
        accumulator2 = _mm256_fmadd_ps(vecData12, vecData22, accumulator2);
    }

    accumulator1 = _mm256_add_ps(accumulator1, accumulator2);
    
    res = horizontal_add(accumulator1);
    
    for (; i < vec1->cols; ++i) {
        res += vec1->data[i] * vec2->data[i];
    }
    
    return res;
}

void matrixMulVecSimd(const Matrix2D *matrix, const Matrix2D *vec, Matrix2D *result) {
    assert(matrix->cols == vec->cols);
    assert(vec->rows == 1);
    if (matrix->cols == 0 || matrix->rows == 0) return;
    Matrix2D row = {.data = NULL, .rows = 1, .cols = matrix->cols};
    for (size_t i = 0; i < matrix->rows; ++i) {
        row.data = MATRIX2D_ROW(*matrix, i);
        result->data[i] = dotSimd(&row, vec);
    }
}


void transposeMatrixSimd(const Matrix2D* src, Matrix2D* dst) {
    assert(dst->rows == src->cols);
    assert(dst->cols == src->rows);
    const size_t simd_width = 8, block_size = 4;
    size_t i, j;
    __m128 row0, row1, row2, row3;
    for (i = 0; i + block_size <= src->rows; i += block_size ) {
        for (j = 0; j + simd_width <= src->cols; j += simd_width) {
                __m256 doubleRow0 = _mm256_loadu_ps(&MATRIX2D_AT(*src, i, j));
                __m256 doubleRow1 = _mm256_loadu_ps(&MATRIX2D_AT(*src, i + 1, j));
                __m256 doubleRow2 = _mm256_loadu_ps(&MATRIX2D_AT(*src, i + 2, j));
                __m256 doubleRow3 = _mm256_loadu_ps(&MATRIX2D_AT(*src, i + 3, j));
                
                // low
                row0 = _mm256_castps256_ps128(doubleRow0);
                row1 = _mm256_castps256_ps128(doubleRow1);
                row2 = _mm256_castps256_ps128(doubleRow2);
                row3 = _mm256_castps256_ps128(doubleRow3);

                _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

                _mm_storeu_ps(&MATRIX2D_AT(*dst, j, i), row0);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 1, i), row1);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 2, i), row2);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 3, i), row3);

                // high
                row0 = _mm256_extractf128_ps(doubleRow0, 1);
                row1 = _mm256_extractf128_ps(doubleRow1, 1);
                row2 = _mm256_extractf128_ps(doubleRow2, 1);
                row3 = _mm256_extractf128_ps(doubleRow3, 1);

                _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 4, i), row0);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 5, i), row1);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 6, i), row2);
                _mm_storeu_ps(&MATRIX2D_AT(*dst, j + 7, i), row3);

        }

        for (; j < src->cols; ++j) {
            MATRIX2D_AT(*dst, j, i) = MATRIX2D_AT(*src, i, j);
            MATRIX2D_AT(*dst, j, i + 1) = MATRIX2D_AT(*src, i, j + 1);
            MATRIX2D_AT(*dst, j , i + 2) = MATRIX2D_AT(*src, i, j + 2);
            MATRIX2D_AT(*dst, j, i + 3) = MATRIX2D_AT(*src, i, j + 3);
        }

    }

        for (; i < src->rows; ++i) {
            for (j = 0; j < src->cols; ++j) {
                MATRIX2D_AT(*dst, j, i) = MATRIX2D_AT(*src, i, j);
            }
        }
}



void _matrixMulMatrixTSimd(const Matrix2D *matrix1, const Matrix2D *matrix2T, Matrix2D *result) {
    assert(matrix1->cols == matrix2T->cols); // shared inner dim
    assert(matrix1->rows == result->rows);
    assert(matrix2T->rows == result->cols); // matrix2T rows = matrix2 cols
    if (matrix1->cols == 0 || matrix1->rows == 0 || matrix2T->rows == 0) return;

    for (size_t i = 0; i < matrix1->rows; ++i) {
        Matrix2D row1 = {.data = MATRIX2D_ROW(*matrix1, i), .rows = 1, .cols = matrix1->cols};
        for (size_t j = 0; j < matrix2T->rows; ++j) {
            Matrix2D row2 = {.data = MATRIX2D_ROW(*matrix2T, j), .rows = 1, .cols = matrix2T->cols};
            MATRIX2D_AT(*result, i, j) = dotSimd(&row1, &row2);
        }
    }
}

 void matrixMulMatrixSimd(const Matrix2D *matrix1, const Matrix2D *matrix2, Matrix2D *result) {
    assert(matrix1->cols == matrix2->rows); // shared inner dim
    assert(matrix1->rows == result->rows);
    assert(matrix2->cols == result->cols); // matrix2T rows = matrix2 cols
    Matrix2D matrix2T = {.data = malloc(matrix2->rows * matrix2->cols * sizeof(float)), .rows = matrix2->cols, .cols = matrix2->rows};
    transposeMatrixSimd(matrix2, &matrix2T);
    _matrixMulMatrixTSimd(matrix1, &matrix2T, result);
 }





void _matmulTileKernelElastic(const float* A, const float* B, float* R,
                 size_t M, size_t N, size_t K,
                 size_t lda, size_t ldb, size_t ldc) {
    const size_t simd_width = 8;
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 a_val = _mm256_set1_ps(A[i * lda + k]);
            for (size_t j = 0; j + simd_width <= N; j += simd_width) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]);  // B[k][j..j+7]
                __m256 r_vec = _mm256_loadu_ps(&R[i * ldc + j]);
                r_vec = _mm256_fmadd_ps(a_val, b_vec, r_vec);
                _mm256_storeu_ps(&R[i * ldc + j], r_vec);
            }
             // Tail scalar fallback
            for (size_t j = N - (N % simd_width); j < N; ++j) {
                R[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

void _matmulTileKernel32(const float* A, const float* B, float* R,
                 size_t lda, size_t ldb, size_t ldc) {
    const size_t simd_width = 8;
    const size_t TILE = simd_width * 4;
    for (size_t i = 0; i < TILE; ++i) {
        __m256 acc0 = _mm256_loadu_ps(&R[i * ldc                ]);
        __m256 acc1 = _mm256_loadu_ps(&R[i * ldc +    simd_width]);
        __m256 acc2 = _mm256_loadu_ps(&R[i * ldc + 2 *simd_width]);
        __m256 acc3 = _mm256_loadu_ps(&R[i * ldc + 3 *simd_width]);
        for (size_t k = 0; k < TILE; ++k) {
            __m256 a = _mm256_set1_ps(A[i * lda + k]);
            __m256 b0 = _mm256_loadu_ps(&B[k * ldb                 ]);
            __m256 b1 = _mm256_loadu_ps(&B[k * ldb +     simd_width]);
            __m256 b2 = _mm256_loadu_ps(&B[k * ldb + 2 * simd_width]);
            __m256 b3 = _mm256_loadu_ps(&B[k * ldb + 3 * simd_width]);

            acc0 = _mm256_fmadd_ps(a, b0, acc0);
            acc1 = _mm256_fmadd_ps(a, b1, acc1);
            acc2 = _mm256_fmadd_ps(a, b2, acc2);
            acc3 = _mm256_fmadd_ps(a, b3, acc3);
        }
        _mm256_storeu_ps(&R[i * ldc                 ], acc0);
        _mm256_storeu_ps(&R[i * ldc +     simd_width], acc1);
        _mm256_storeu_ps(&R[i * ldc + 2 * simd_width], acc2);
        _mm256_storeu_ps(&R[i * ldc + 3 * simd_width], acc3);
    }
}

void _matmulTileKernel64(const float* A, const float* B, float* R,
                 size_t lda, size_t ldb, size_t ldc) {
    const size_t simd_width = 8;
    const size_t TILE = simd_width * 8;
    for (size_t i = 0; i < TILE; ++i) {
        __m256 acc0 = _mm256_loadu_ps(&R[i * ldc                ]);
        __m256 acc1 = _mm256_loadu_ps(&R[i * ldc +     simd_width]);
        __m256 acc2 = _mm256_loadu_ps(&R[i * ldc + 2 * simd_width]);
        __m256 acc3 = _mm256_loadu_ps(&R[i * ldc + 3 * simd_width]);
        __m256 acc4 = _mm256_loadu_ps(&R[i * ldc + 4 * simd_width]);
        __m256 acc5 = _mm256_loadu_ps(&R[i * ldc + 5 * simd_width]);
        __m256 acc6 = _mm256_loadu_ps(&R[i * ldc + 6 * simd_width]);
        __m256 acc7 = _mm256_loadu_ps(&R[i * ldc + 7 * simd_width]);
        
        for (size_t k = 0; k < TILE; ++k) {
            __m256 a = _mm256_set1_ps(A[i * lda + k]);
            __m256 b0 = _mm256_loadu_ps(&B[k * ldb                 ]);
            __m256 b1 = _mm256_loadu_ps(&B[k * ldb +     simd_width]);
            __m256 b2 = _mm256_loadu_ps(&B[k * ldb + 2 * simd_width]);
            __m256 b3 = _mm256_loadu_ps(&B[k * ldb + 3 * simd_width]);
            __m256 b4 = _mm256_loadu_ps(&B[k * ldb + 4 * simd_width]);
            __m256 b5 = _mm256_loadu_ps(&B[k * ldb + 5 * simd_width]);
            __m256 b6 = _mm256_loadu_ps(&B[k * ldb + 6 * simd_width]);
            __m256 b7 = _mm256_loadu_ps(&B[k * ldb + 7 * simd_width]);

            acc0 = _mm256_fmadd_ps(a, b0, acc0);
            acc1 = _mm256_fmadd_ps(a, b1, acc1);
            acc2 = _mm256_fmadd_ps(a, b2, acc2);
            acc3 = _mm256_fmadd_ps(a, b3, acc3);
            acc4 = _mm256_fmadd_ps(a, b4, acc4);
            acc5 = _mm256_fmadd_ps(a, b5, acc5);
            acc6 = _mm256_fmadd_ps(a, b6, acc6);
            acc7 = _mm256_fmadd_ps(a, b7, acc7);
        }
        
        _mm256_storeu_ps(&R[i * ldc                 ], acc0);
        _mm256_storeu_ps(&R[i * ldc +     simd_width], acc1);
        _mm256_storeu_ps(&R[i * ldc + 2 * simd_width], acc2);
        _mm256_storeu_ps(&R[i * ldc + 3 * simd_width], acc3);
        _mm256_storeu_ps(&R[i * ldc + 4 * simd_width], acc4);
        _mm256_storeu_ps(&R[i * ldc + 5 * simd_width], acc5);
        _mm256_storeu_ps(&R[i * ldc + 6 * simd_width], acc6);
        _mm256_storeu_ps(&R[i * ldc + 7 * simd_width], acc7);
    }
}

void _matmulTileKernel128(const float* A, const float* B, float* R,
                 size_t lda, size_t ldb, size_t ldc) {
    const size_t simd_width = 8;
    const size_t TILE = simd_width * 16;
    for (size_t i = 0; i < TILE; ++i) {
        __m256 acc0 = _mm256_loadu_ps(&R[i * ldc                  ]);
        __m256 acc1 = _mm256_loadu_ps(&R[i * ldc +     simd_width]);
        __m256 acc2 = _mm256_loadu_ps(&R[i * ldc + 2 * simd_width]);
        __m256 acc3 = _mm256_loadu_ps(&R[i * ldc + 3 * simd_width]);
        __m256 acc4 = _mm256_loadu_ps(&R[i * ldc + 4 * simd_width]);
        __m256 acc5 = _mm256_loadu_ps(&R[i * ldc + 5 * simd_width]);
        __m256 acc6 = _mm256_loadu_ps(&R[i * ldc + 6 * simd_width]);
        __m256 acc7 = _mm256_loadu_ps(&R[i * ldc + 7 * simd_width]);
        __m256 acc8 = _mm256_loadu_ps(&R[i * ldc + 8 * simd_width]);
        __m256 acc9 = _mm256_loadu_ps(&R[i * ldc + 9 * simd_width]);
        __m256 acc10 = _mm256_loadu_ps(&R[i * ldc + 10 * simd_width]);
        __m256 acc11 = _mm256_loadu_ps(&R[i * ldc + 11 * simd_width]);
        __m256 acc12 = _mm256_loadu_ps(&R[i * ldc + 12 * simd_width]);
        __m256 acc13 = _mm256_loadu_ps(&R[i * ldc + 13 * simd_width]);
        __m256 acc14 = _mm256_loadu_ps(&R[i * ldc + 14 * simd_width]);
        __m256 acc15 = _mm256_loadu_ps(&R[i * ldc + 15 * simd_width]);
        
        for (size_t k = 0; k < TILE; ++k) {
            __m256 a = _mm256_set1_ps(A[i * lda + k]);
            __m256 b0 = _mm256_loadu_ps(&B[k * ldb                  ]);
            __m256 b1 = _mm256_loadu_ps(&B[k * ldb +     simd_width]);
            __m256 b2 = _mm256_loadu_ps(&B[k * ldb + 2 * simd_width]);
            __m256 b3 = _mm256_loadu_ps(&B[k * ldb + 3 * simd_width]);
            __m256 b4 = _mm256_loadu_ps(&B[k * ldb + 4 * simd_width]);
            __m256 b5 = _mm256_loadu_ps(&B[k * ldb + 5 * simd_width]);
            __m256 b6 = _mm256_loadu_ps(&B[k * ldb + 6 * simd_width]);
            __m256 b7 = _mm256_loadu_ps(&B[k * ldb + 7 * simd_width]);
            __m256 b8 = _mm256_loadu_ps(&B[k * ldb + 8 * simd_width]);
            __m256 b9 = _mm256_loadu_ps(&B[k * ldb + 9 * simd_width]);
            __m256 b10 = _mm256_loadu_ps(&B[k * ldb + 10 * simd_width]);
            __m256 b11 = _mm256_loadu_ps(&B[k * ldb + 11 * simd_width]);
            __m256 b12 = _mm256_loadu_ps(&B[k * ldb + 12 * simd_width]);
            __m256 b13 = _mm256_loadu_ps(&B[k * ldb + 13 * simd_width]);
            __m256 b14 = _mm256_loadu_ps(&B[k * ldb + 14 * simd_width]);
            __m256 b15 = _mm256_loadu_ps(&B[k * ldb + 15 * simd_width]);

            acc0 = _mm256_fmadd_ps(a, b0, acc0);
            acc1 = _mm256_fmadd_ps(a, b1, acc1);
            acc2 = _mm256_fmadd_ps(a, b2, acc2);
            acc3 = _mm256_fmadd_ps(a, b3, acc3);
            acc4 = _mm256_fmadd_ps(a, b4, acc4);
            acc5 = _mm256_fmadd_ps(a, b5, acc5);
            acc6 = _mm256_fmadd_ps(a, b6, acc6);
            acc7 = _mm256_fmadd_ps(a, b7, acc7);
            acc8 = _mm256_fmadd_ps(a, b8, acc8);
            acc9 = _mm256_fmadd_ps(a, b9, acc9);
            acc10 = _mm256_fmadd_ps(a, b10, acc10);
            acc11 = _mm256_fmadd_ps(a, b11, acc11);
            acc12 = _mm256_fmadd_ps(a, b12, acc12);
            acc13 = _mm256_fmadd_ps(a, b13, acc13);
            acc14 = _mm256_fmadd_ps(a, b14, acc14);
            acc15 = _mm256_fmadd_ps(a, b15, acc15);
        }
        
        _mm256_storeu_ps(&R[i * ldc                  ], acc0);
        _mm256_storeu_ps(&R[i * ldc +     simd_width], acc1);
        _mm256_storeu_ps(&R[i * ldc + 2 * simd_width], acc2);
        _mm256_storeu_ps(&R[i * ldc + 3 * simd_width], acc3);
        _mm256_storeu_ps(&R[i * ldc + 4 * simd_width], acc4);
        _mm256_storeu_ps(&R[i * ldc + 5 * simd_width], acc5);
        _mm256_storeu_ps(&R[i * ldc + 6 * simd_width], acc6);
        _mm256_storeu_ps(&R[i * ldc + 7 * simd_width], acc7);
        _mm256_storeu_ps(&R[i * ldc + 8 * simd_width], acc8);
        _mm256_storeu_ps(&R[i * ldc + 9 * simd_width], acc9);
        _mm256_storeu_ps(&R[i * ldc + 10 * simd_width], acc10);
        _mm256_storeu_ps(&R[i * ldc + 11 * simd_width], acc11);
        _mm256_storeu_ps(&R[i * ldc + 12 * simd_width], acc12);
        _mm256_storeu_ps(&R[i * ldc + 13 * simd_width], acc13);
        _mm256_storeu_ps(&R[i * ldc + 14 * simd_width], acc14);
        _mm256_storeu_ps(&R[i * ldc + 15 * simd_width], acc15);
    }
}


 void matrixMulMatrixTiled(const Matrix2D *matrix1, const Matrix2D *matrix2, Matrix2D *result) {
    assert(matrix1->cols == matrix2->rows); // shared inner dim
    assert(matrix1->rows == result->rows);
    assert(matrix2->cols == result->cols); // matrix2T rows = matrix2 cols
    size_t TILE;
    void (*kernel)(const float*, const float*, float*, size_t, size_t, size_t);
    if (matrix1->cols >= 1024) {
        kernel = &_matmulTileKernel128;
        TILE = 128;
    } else if (matrix1-> cols >= 256) {
        TILE = 64;
        kernel = &_matmulTileKernel64;
    } else {
        TILE = 32;
        kernel = &_matmulTileKernel32;
    }
    omp_set_num_threads(8);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < matrix1->rows; i += TILE) {
        for (size_t j = 0; j < matrix2->cols; j += TILE) {
            size_t M = i + TILE > matrix1->rows ? matrix1->rows - i : TILE;
            size_t N = (j + TILE > matrix2->cols) ? (matrix2->cols - j) : TILE;
            for (size_t k = 0; k < matrix1->cols; k += TILE) {
                size_t K = (k + TILE > matrix1->cols) ? (matrix1->cols - k) : TILE;
                if (M == TILE && N == TILE && K == TILE) {
                    kernel(&MATRIX2D_AT(*matrix1, i, k), &MATRIX2D_AT(*matrix2, k, j), &MATRIX2D_AT(*result,i, j), matrix1->cols, matrix2->cols, result->cols);
                } else {
                    _matmulTileKernelElastic(&MATRIX2D_AT(*matrix1, i, k), &MATRIX2D_AT(*matrix2, k, j), &MATRIX2D_AT(*result,i, j), M, N, K, matrix1->cols, matrix2->cols, result->cols);
                }
            }
        }
    }
 }