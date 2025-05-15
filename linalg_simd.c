#include "linalg.h"
#include <immintrin.h>
#include <string.h>

static inline float horizontal_add(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void scalarMulVecSimd(const float scalar, const Mat *vec, Mat *result) {
    const size_t simd_width = 8;
    size_t i;
    
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    for (i = 0; i + simd_width <= vec->rows * vec->cols; i += simd_width) {
        __m256 vec_data = _mm256_loadu_ps(&vec->data[i]);
        __m256 mul_result = _mm256_mul_ps(vec_data, scalar_vec);
        _mm256_storeu_ps(&result->data[i], mul_result);
    }
    
    for (; i < vec->rows * vec->cols; i++) {
        result->data[i] = scalar * vec->data[i];
    }
}

float dotSimd(const Mat *vec1, const Mat *vec2) {
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
    
    for (; i < vec1->cols; i++) {
        res += vec1->data[i] * vec2->data[i];
    }
    
    return res;
}

void matrixMulVecSimd(const Mat *matrix, const Mat *vec, Mat *result) {
    
    Mat row = {.data = NULL, .rows = 1, .cols = matrix->cols};
    for (size_t i = 0; i < matrix->rows; i++) {
        row.data = MAT_ROW(*matrix, i);
        result->data[i] = dotSimd(&row, vec);
    }
}


