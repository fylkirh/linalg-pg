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

void matrixMulMatrixSimd(const Matrix2D *matrix1, const Matrix2D *matrix2T, Matrix2D *result) {
    assert(matrix1->cols == matrix2T->cols); // shared inner dim
    assert(matrix1->rows == result->rows);
    assert(matrix2T->rows == result->cols); // matrix2T rows = matrix2 cols
    if (matrix1->cols == 0 || matrix1->rows == 0 || matrix2T->rows == 0) return;
    Matrix2D row1 = {.data = NULL, .rows = 1, .cols = matrix1->cols};
    Matrix2D row2 = {.data = NULL, .rows = 1, .cols = matrix2T->cols};
    for (size_t i = 0; i < matrix1->rows; ++i) {
        row1.data = MATRIX2D_ROW(*matrix1, i);
        for (size_t j = 0; j < matrix2T->rows; ++j) {
            row2.data = MATRIX2D_ROW(*matrix2T, j);
            MATRIX2D_AT(*result, i, j) = dotSimd(&row1, &row2);
        }
    }
}

