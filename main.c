#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"

#define SIZE 500000
#define MATRIX_SIZE 4000
#define MATRIX_MUL_SIZE 1000  // Smaller size for matrix-matrix multiplication

void test_scalar_mul(const char* name, void (*func)(const float, const Matrix2D*, Matrix2D*)) {
    Matrix2D vec = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    Matrix2D result = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    float scalar = 2.0f;
    
    // Initialize input vector
    for (size_t i = 0; i < SIZE; i++) {
        vec.data[i] = (float)(i + 1);
    }
    
    // Measure time
    clock_t start = clock();
    func(scalar, &vec, &result);
    clock_t end = clock();
    
    // Verify first few results
    printf("%s implementation:\n", name);
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", result.data[i]);
    }
    printf("\n");
    
    // Print timing
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n\n", time_taken);
    
    free(vec.data);
    free(result.data);
}

void test_dot(const char* name, float (*func)(const Matrix2D*, const Matrix2D*)) {
    Matrix2D vec1 = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    Matrix2D vec2 = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    
    // Initialize input vectors
    for (size_t i = 0; i < SIZE; i++) {
        vec1.data[i] = (float)(i + 1);
        vec2.data[i] = (float)(i + 1);
    }
    
    // Measure time
    clock_t start = clock();
    float result = func(&vec1, &vec2);
    clock_t end = clock();
    
    printf("%s implementation:\n", name);
    printf("Result: %f\n", result);
    
    // Print timing
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n\n", time_taken);
    
    free(vec1.data);
    free(vec2.data);
}

void compare_matrix_mul_implementations() {
    // Allocate matrices
    Matrix2D matrix = {
        .data = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float)),
        .rows = MATRIX_SIZE,
        .cols = MATRIX_SIZE
    };
    Matrix2D vec = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    Matrix2D result_plain = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    Matrix2D result_simd = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    
    // Initialize matrix and vector
    for (size_t i = 0; i < MATRIX_SIZE; i++) {
        for (size_t j = 0; j < MATRIX_SIZE; j++) {
            MATRIX2D_AT(matrix, i, j) = (float)((i + j) % 10);
        }
        vec.data[i] = (float)(i % 5);
    }
    
    // Test plain implementation
    clock_t start_plain = clock();
    matrixMulVec(&matrix, &vec, &result_plain);
    clock_t end_plain = clock();
    double time_plain = ((double)(end_plain - start_plain)) / CLOCKS_PER_SEC;
    
    // Test SIMD implementation
    clock_t start_simd = clock();
    matrixMulVecSimd(&matrix, &vec, &result_simd);
    clock_t end_simd = clock();
    double time_simd = ((double)(end_simd - start_simd)) / CLOCKS_PER_SEC;
    
    // Print comparison
    printf("Matrix-Vector Multiplication Comparison (size: %dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Plain implementation: %f seconds\n", time_plain);
    printf("SIMD implementation:  %f seconds\n", time_simd);
    printf("Speedup: %.2fx\n\n", time_plain / time_simd);
    
    // Verify results match
    int results_match = 1;
    for (size_t i = 0; i < MATRIX_SIZE; i++) {
        if (fabs(result_plain.data[i] - result_simd.data[i]) > 1e-6) {
            results_match = 0;
            printf("Results differ at index %zu: %f vs %f\n", i, result_plain.data[i], result_simd.data[i]);
            break;
        }
    }
    printf("Results %s\n\n", results_match ? "match" : "differ");
    
    // Free allocated memory
    free(matrix.data);
    free(vec.data);
    free(result_plain.data);
    free(result_simd.data);
}

void compare_matrix_mul_matrix_implementations() {
    // Define matrix dimensions
    const size_t m = MATRIX_MUL_SIZE;      // rows of first matrix
    const size_t k = MATRIX_MUL_SIZE/2;    // cols of first matrix / rows of second matrix
    const size_t n = MATRIX_MUL_SIZE*2;    // cols of second matrix
    
    // Allocate matrices
    Matrix2D matrix1 = {
        .data = malloc(m * k * sizeof(float)),
        .rows = m,
        .cols = k
    };
    Matrix2D matrix2 = {
        .data = malloc(k * n * sizeof(float)),
        .rows = k,
        .cols = n
    };
    Matrix2D result_plain = {
        .data = malloc(m * n * sizeof(float)),
        .rows = m,
        .cols = n
    };
    Matrix2D result_simd = {
        .data = malloc(m * n * sizeof(float)),
        .rows = m,
        .cols = n
    };

    Matrix2D result_tiled = {
        .data = malloc(m * n * sizeof(float)),
        .rows = m,
        .cols = n
    };
    
    // Initialize matrices
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            MATRIX2D_AT(matrix1, i, j) = (float)((i + j) % 10);
        }
    }
    // Initialize matrix2 in normal form
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            MATRIX2D_AT(matrix2, i, j) = (float)((i * j) % 10);
        }
    }
    
    // Test plain implementation
    clock_t start_plain = clock();
    matrixMulMatrix(&matrix1, &matrix2, &result_plain);
    clock_t end_plain = clock();
    double time_plain = ((double)(end_plain - start_plain)) / CLOCKS_PER_SEC;
    
    // Test SIMD implementation
    clock_t start_simd = clock();
    matrixMulMatrixSimd(&matrix1, &matrix2, &result_simd);
    clock_t end_simd = clock();
    double time_simd = ((double)(end_simd - start_simd)) / CLOCKS_PER_SEC;

    // Test tiled implementation
    clock_t start_tiled = clock();
    matrixMulMatrixTiled(&matrix1, &matrix2, &result_tiled);
    clock_t end_tiled = clock();
    double time_tiled = ((double)(end_tiled - start_tiled)) / CLOCKS_PER_SEC;
    
    // Print comparison
    printf("Matrix-Matrix Multiplication Comparison:\n");
    printf("Matrix1: %zux%zu, Matrix2: %zux%zu, Result: %zux%zu\n", 
           m, k, k, n, m, n);
    printf("Plain implementation: %f seconds\n", time_plain);
    printf("SIMD implementation:  %f seconds\n", time_simd);
    printf("Tiled implementation:  %f seconds\n", time_tiled);
    printf("Speedup simd: %.2fx\n", time_plain / time_simd);
    printf("Speedup tiled: %.2fx\n\n", time_plain / time_tiled);
    
    // Verify results match
    int results_match = 1;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if (fabs(MATRIX2D_AT(result_plain, i, j) - MATRIX2D_AT(result_simd, i, j)) > 1e-6) {
                results_match = 0;
                printf("Results differ at index (%zu,%zu): %f vs %f\n", 
                       i, j, MATRIX2D_AT(result_plain, i, j), MATRIX2D_AT(result_simd, i, j));
                goto end_verification;  // Break out of nested loops
            }
        }
    }
end_verification:
    printf("Results %s\n\n", results_match ? "match" : "differ");
    
    // Free allocated memory
    free(matrix1.data);
    free(matrix2.data);
    free(result_plain.data);
    free(result_simd.data);
}

int main() {
    printf("Testing vector scalar multiplication with size %d\n\n", SIZE);
    
    test_scalar_mul("Plain", scalarMulVec);
    test_scalar_mul("SIMD", scalarMulVecSimd);
    
    printf("Testing dot product with size %d\n\n", SIZE);
    test_dot("Plain", dot);
    test_dot("SIMD", dotSimd);
    
    printf("Testing matrix-vector multiplication:\n");
    compare_matrix_mul_implementations();
    
    printf("Testing matrix-matrix multiplication:\n");
    compare_matrix_mul_matrix_implementations();
    
    return 0;
} 