#include "linalg.h"
#include <immintrin.h>
#include <assert.h>

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
    assert(vec1->rows == vec2->rows == 1);
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