#include <cuda_runtime_api.h>

__global__ void scaleKernel(float *src, float *dst, float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * scale;
}

int main()
{

    float *a_dev;
    float *b_dev;
    float *a = new float[128*100];
    float *b = new float[128*100];
    cudaSetDevice(0); 

    cudaMalloc(&a_dev, sizeof(float)*128*100);
    cudaMalloc(&b_dev, sizeof(float)*128*100);

    cudaMemcpy (a_dev, a, sizeof(float)*1000, cudaMemcpyHostToDevice);

    scaleKernel<<<100, 128>>>(a_dev, b_dev, 4.0f);
    scaleKernel<<<50, 128>>>(a_dev, b_dev, 4.0f);
    scaleKernel<<<50, 128>>>(a_dev+50, b_dev+50, 4.0f);
    scaleKernel<<<100, 128>>>(a_dev, b_dev, 4.0f);
    cudaDeviceSynchronize();

    cudaMemcpy (b, b_dev, sizeof(int)*1000, cudaMemcpyDeviceToHost);
}