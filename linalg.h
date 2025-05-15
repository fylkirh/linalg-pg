#include <stddef.h>

typedef struct {
    float* data;        // flat array
    size_t rows;
    size_t cols;
} Mat;

#define MAT_AT(mat, i, j) ((mat).data[(i) * (mat).cols + (j)])
#define MAT_ROW(mat, i) (&((mat).data[(i) * (mat).cols]))

void scalarMulVec(const float scalar, const Mat *vec, Mat *result);
void scalarMulVecSimd(const float scalar, const Mat *vec, Mat *result);

float dot(const Mat *vec1, const Mat *vec2);
float dotSimd(const Mat *vec1, const Mat *vec2);

void matrixMulVec(const Mat *matrix, const Mat *vec, Mat *result);
void matrixMulVecSimd(const Mat *matrix, const Mat *vec, Mat *result);