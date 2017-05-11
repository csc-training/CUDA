#include <cuda_runtime_api.h>

__global__ void scaleKernel(float *src, float *dst, float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * scale;
}

int main()
{
    float *a;
    float *b;
    cudaSetDevice(0); 

    cudaMalloc(&a, sizeof(float)*100*128);
    cudaMalloc(&b, sizeof(float)*100*128);

    copyKernel<<<100, 128>>>(a, b, 4.0f);
    copyKernel<<<50, 128>>>(a, b, 4.0f);
    copyKernel<<<50, 128>>>(a+50, b+50, 4.0f);
    copyKernel<<<100, 128>>>(a, b, 4.0f);
    cudaDeviceSynchronize();
}