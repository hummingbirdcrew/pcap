#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Corrected the function signature to accept the array size 'n'
__global__ void vectoraddthreadn(int* a, int* b, int* r, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // Use correct variable 'i' to access elements
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

// Corrected the function signature to accept the array size 'n'
__global__ void vectoraddblockn(int* a, int* b, int* r, int n) {
    int i = threadIdx.x;
    // Use correct variable 'i' to access elements
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

int main() {
    int* a, * b, * c; // Changed 'r' to 'c'
    int n=7;
    a = (int*)malloc(n * sizeof(int));
    b = (int*)malloc(n * sizeof(int));
    c = (int*)malloc(n * sizeof(int)); // Changed 'r' to 'c'
    for (int i = 0; i <n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }
    printf("pirnting a");
    for (int i = 0; i <n; ++i) {
        printf("%d",a[i]);
        printf("\n");

    }
    printf("pirnting b");
    for (int i = 0; i <n; ++i) {
        printf("%d",b[i]);
        printf("\n");

    }

    int* d_a, * d_b, * d_c; // Changed 'd_r' to 'd_c'
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int)); // Changed 'd_r' to 'd_c'

    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Pass the size 'n' to the kernel function
    vectoraddblockn<<<1, n>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result using block size as N:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n\n");

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Pass the size 'n' to the kernel function
    vectoraddthreadn<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result using N threads:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free allocated memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free allocated memory on the host
    free(a);
    free(b);
    free(c);

    return 0;
}
