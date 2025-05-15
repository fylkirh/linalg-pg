#include <stddef.h>

typedef struct {
    float* data;        // flat array
    size_t rows;
    size_t cols;
} Matrix2D;

#define MATRIX2D_AT(mat, i, j) ((mat).data[(i) * (mat).cols + (j)])
#define MATRIX2D_ROW(mat, i) (&((mat).data[(i) * (mat).cols]))

void scalarMulVec(const float scalar, const Matrix2D *vec, Matrix2D *result);
void scalarMulVecSimd(const float scalar, const Matrix2D *vec, Matrix2D *result);

float dot(const Matrix2D *vec1, const Matrix2D *vec2);
float dotSimd(const Matrix2D *vec1, const Matrix2D *vec2);

void matrixMulVec(const Matrix2D *matrix, const Matrix2D *vec, Matrix2D *result);
void matrixMulVecSimd(const Matrix2D *matrix, const Matrix2D *vec, Matrix2D *result);