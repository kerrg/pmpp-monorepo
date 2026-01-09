#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Computes vector sum C = A + B on the GPU
// Each thread performs one pair-wise addition
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i]; //TODO: so many questions about overflows
  }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d = NULL;

  // In the C++-ification version, add some wrappers to check for errors and abort.
  cudaMalloc((void **) &A_d, size);
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
    // --- Existing 16-element test ---
    int N = 16;
    float A[N], B[N], C[N];

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }

    // Call the wrapper function
    vecAdd(A, B, C, N);

    // Pretty print the result
    printf("Vector C (Result):\n[ ");
    for (int i = 0; i < N; i++) {
        printf("%.1f", C[i]);
        if (i < N - 1) printf(", ");
    }
    printf(" ]\n\n");
}
