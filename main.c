#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"

#define SIZE 500000
#define MATRIX_SIZE 30000

void test_scalar_mul(const char* name, void (*func)(const float, const Mat*, Mat*)) {
    Mat vec = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    Mat result = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
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

void test_dot(const char* name, float (*func)(const Mat*, const Mat*)) {
    Mat vec1 = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    Mat vec2 = {.data = malloc(SIZE * sizeof(float)), .rows = 1, .cols = SIZE};
    
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
    Mat matrix = {
        .data = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float)),
        .rows = MATRIX_SIZE,
        .cols = MATRIX_SIZE
    };
    Mat vec = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    Mat result_plain = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    Mat result_simd = {
        .data = malloc(MATRIX_SIZE * sizeof(float)),
        .rows = 1,
        .cols = MATRIX_SIZE
    };
    
    // Initialize matrix and vector
    for (size_t i = 0; i < MATRIX_SIZE; i++) {
        for (size_t j = 0; j < MATRIX_SIZE; j++) {
            MAT_AT(matrix, i, j) = (float)((i + j) % 10);
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

int main() {
    printf("Testing vector scalar multiplication with size %d\n\n", SIZE);
    
    test_scalar_mul("Plain", scalarMulVec);
    test_scalar_mul("SIMD", scalarMulVecSimd);
    
    printf("Testing dot product with size %d\n\n", SIZE);
    test_dot("Plain", dot);
    test_dot("SIMD", dotSimd);
    printf("Testing matrix-vector multiplication:\n");
    compare_matrix_mul_implementations();
    
    return 0;
} 