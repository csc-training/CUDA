#include <cuda_runtime_api.h>
#include <iostream>

__global__ void copyKernel(int *src, int *dst, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > size)
        return;
    dst[idx] = src[idx];
}

int main()
{
    int *a_dev;
    int *b_dev;
    int *a = new int[1000];
    int *b = new int[1000];
    cudaSetDevice(0); 

    cudaMalloc(&a_dev, sizeof(int)*1000);
    cudaMalloc(&b_dev, sizeof(int)*1000);

    cudaMemcpy (a_dev, a, sizeof(int)*1000, cudaMemcpyHostToDevice);

    copyKernel<<<100, 128>>>(a_dev, b_dev, 1000);

    cudaMemcpy (b, b_dev, sizeof(int)*1000, cudaMemcpyDeviceToHost);

}
