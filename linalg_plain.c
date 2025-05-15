#include "linalg.h"
#include <assert.h>

void scalarMulVec(const float scalar, const Matrix2D *vec, Matrix2D *result) {
    for (size_t i = 0; i < vec->rows * vec->cols; ++i) {
        result->data[i] = scalar * vec->data[i];
    }
}

float dot(const Matrix2D *vec1, const Matrix2D *vec2) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec1->cols; ++i) {
        sum += vec1->data[i] * vec2->data[i];
    }
    return sum;
}

void matrixMulVec(const Matrix2D *matrix, const Matrix2D *vec, Matrix2D *result) {
    for (size_t i = 0; i < matrix->rows; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < matrix->cols; j++) {
            sum += MATRIX2D_AT(*matrix, i, j) * vec->data[j];
        }
        result->data[i] = sum;
    }
}

void matrixMulMatrix(const Matrix2D *matrix1, const Matrix2D *matrix2, Matrix2D *result) {
    assert(matrix1->cols == matrix2->rows);
    for (size_t i = 0; i < matrix1->rows; ++i) {
        for (size_t j = 0; j < matrix2->cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < matrix1->cols; ++k) {
                sum += (MATRIX2D_AT(*matrix1, i, k)) * (MATRIX2D_AT(*matrix2, k, j));
            }
            MATRIX2D_AT(*result, i, j) = sum;
        }

    }
}