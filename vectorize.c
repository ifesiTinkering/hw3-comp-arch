#include <stdio.h>
#include <immintrin.h>  // Header for AVX intrinsics
#include <time.h>

#define VECTOR_SIZE 1024

void multiply_vectors_baseline(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

void multiply_vectors_vectorized(float* a, float* b, float* result, int size) {
    int i;
    for (i = 0; i <= size - 8; i += 8) {  // Process 8 elements at a time
        __m256 vec_a = _mm256_loadu_ps(&a[i]);  // Load 8 floats from array a
        __m256 vec_b = _mm256_loadu_ps(&b[i]);  // Load 8 floats from array b
        __m256 vec_result = _mm256_mul_ps(vec_a, vec_b);  // Perform vectorized multiplication
        _mm256_storeu_ps(&result[i], vec_result);  // Store result back to array
    }
    // Handle remaining elements
    for (; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

int main() {
    float a[VECTOR_SIZE], b[VECTOR_SIZE], result_baseline[VECTOR_SIZE], result_vectorized[VECTOR_SIZE];
    clock_t start, end;
    double time_baseline, time_vectorized;

    // Initialize input arrays
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = (float)i;
        b[i] = (float)(i + 1);
    }

    // Baseline multiplication
    start = clock();
    multiply_vectors_baseline(a, b, result_baseline, VECTOR_SIZE);
    end = clock();
    time_baseline = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Baseline multiplication time: %f seconds\n", time_baseline);

    // Vectorized multiplication
    start = clock();
    multiply_vectors_vectorized(a, b, result_vectorized, VECTOR_SIZE);
    end = clock();
    time_vectorized = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Vectorized multiplication time: %f seconds\n", time_vectorized);

    // Compare results
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (result_baseline[i] != result_vectorized[i]) {
            printf("Mismatch at index %d: baseline = %f, vectorized = %f\n", i, result_baseline[i], result_vectorized[i]);
            return 1;
        }
    }

    printf("Results match for all elements.\n");
    return 0;
}
