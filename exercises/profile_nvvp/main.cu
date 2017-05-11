#include <cuda_runtime_api.h>

__global__ void copyKernel(int *src, int *dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}

int main()
{
    int *a;
    int *b;
    cudaSetDevice(0); 

    cudaMalloc(&a, sizeof(int)*100*128);
    cudaMalloc(&b, sizeof(int)*100*128);

    copyKernel<<<100, 128>>>(a, b);
    cudaDeviceSynchronize();
}