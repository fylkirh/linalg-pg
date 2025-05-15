#include "linalg.h"

void scalarMulVec(const float scalar, const Mat *vec, Mat *result) {
    for (size_t i = 0; i < vec->rows * vec->cols; i++) {
        result->data[i] = scalar * vec->data[i];
    }
}

float dot(const Mat *vec1, const Mat *vec2) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec1->cols; i++) {
        sum += vec1->data[i] * vec2->data[i];
    }
    return sum;
}

void matrixMulVec(const Mat *matrix, const Mat *vec, Mat *result) {
    for (size_t i = 0; i < matrix->rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < matrix->cols; j++) {
            sum += MAT_AT(*matrix, i, j) * vec->data[j];
        }
        result->data[i] = sum;
    }
}