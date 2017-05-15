#include <cuda_runtime_api.h>

__global__ void copyKernel(int *src, int *dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}

int main()
{
    int *a_dev;
    int *b_dev;
    int *a = new int[128*100];
    int *b = new int[128*100];
    cudaSetDevice(0); 

    cudaMalloc(&a_dev, sizeof(int)*128*100);
    cudaMalloc(&b_dev, sizeof(int)*128*100);

    cudaMemcpy (a_dev, a, sizeof(int)*1000, cudaMemcpyHostToDevice);

    copyKernel<<<100, 128>>>(a_dev, b_dev);
    cudaDeviceSynchronize();

    cudaMemcpy (b, b_dev, sizeof(int)*1000, cudaMemcpyDeviceToHost);

}