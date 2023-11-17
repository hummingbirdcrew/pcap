###Implement a CUDA program to add two vectors of length N by keeping the number of threads per block as 256 (constant) and vary the number of blocks to handle N elements


%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectoradd(int* a,int* b,int* c,int n){
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
    if(tid<n){
    c[tid]=a[tid]+b[tid];
    }
}

int main(){
    int *a,*b,*c;
    int *da,*db,*dc;
    int n=7;
    a=(int*)malloc(n*sizeof(int));
    b=(int*)malloc(n*sizeof(int));
    c=(int*)malloc(n*sizeof(int));

    for (int i = 0; i <n; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }
    printf("pirnting a\n");
    for (int i = 0; i <n; ++i) {
        printf("%d",a[i]);
        printf("\n");

    }
    printf("pirnting b\n");
    for (int i = 0; i <n; ++i) {
        printf("%d",b[i]);
        printf("\n");

    }

    cudaMalloc((void**)&da,n*sizeof(int));
    cudaMalloc((void**)&db,n*sizeof(int));
    cudaMalloc((void**)&dc,n*sizeof(int));

    cudaMemcpy(da,a,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,n*sizeof(int),cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (n +255) /256;

    vectoradd<<<numBlocks,256>>>(da,db,dc,n);

    cudaMemcpy(c,dc,n*sizeof(int),cudaMemcpyDeviceToHost);

    printf("Result using N threads:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free allocated memory on the device
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // Free allocated memory on the host
    free(a);
    free(b);
    free(c);

    return 0;
}
