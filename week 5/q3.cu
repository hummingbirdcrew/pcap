#####third question
Write a program in CUDA which performs convolution operation on one dimensional input array N of size width using a mask array M of size mask_width to produce the resultant one dimensional array P of size width


%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolution(int* p, int* m, int* r, int ml, int pl) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int rvalue = 0;
    int pstart = tid - (ml) / 2;
    for (int j = 0; j < ml; j++) {
        if ((pstart + j) >= 0 && (pstart + j) < pl) {
            rvalue += p[pstart + j] * m[j];
        }
    }
    r[tid] = rvalue;
}

int main() {
    int *m, *n, *r;
    int *dm, *dn, *dr;
    int ns = 7;
    int ms = 5;

    m = (int*)malloc(ms * sizeof(int));
    n = (int*)malloc(ns * sizeof(int));
    r = (int*)malloc(ns * sizeof(int));

    for (int i = 0; i < ns; ++i) {
        n[i] = i;
    }
    for (int i = 0; i < ms; ++i) {
        m[i] = i * 2;
    }
    printf("printing array n\n");
    for (int i = 0; i < ns; ++i) {
        printf("%d ", n[i]);
    }
    printf("\n");

    printf("printing mask m\n");
    for (int i = 0; i < ms; ++i) {
        printf("%d ", m[i]);
    }
    printf("\n");

    cudaMalloc((void**)&dn, ns * sizeof(int));
    cudaMalloc((void**)&dm, ms * sizeof(int));
    cudaMalloc((void**)&dr, ns * sizeof(int));

    cudaMemcpy(dn, n, ns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dm, m, ms * sizeof(int), cudaMemcpyHostToDevice);

    convolution<<<(ns + 4 - 1) / 4, 4>>>(dn, dm, dr, ms, ns);

    cudaMemcpy(r, dr, ns * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result after convolution:\n");
    for (int i = 0; i < ns; ++i) {
        printf("%d ", r[i]);
    }
    printf("\n");

    cudaFree(dn);
    cudaFree(dm);
    cudaFree(dr);
    free(m);
    free(n);
    free(r);

    return 0;
}
