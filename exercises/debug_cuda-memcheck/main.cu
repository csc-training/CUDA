#include <cuda_runtime_api.h>


#define getErrorCuda(command)\
		command;\
		cudaDeviceSynchronize();\
		if (cudaPeekAtLastError() != cudaSuccess){\
			std::cout << #command << " : " << cudaGetErrorString(cudaGetLastError())\
			 << " in file " << __FILE__ << " at line " << __LINE__ << std::endl;\
			exit(1);\
		}


__global__ void copyKernel(int *src, int *dst, int scale, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > size)
        return;
    dst[idx] = src[idx];
}

int main()
{
    int *a;
    int *b;
    cudaSetDevice(0); 

    cudaMalloc(&a, sizeof(int)*1000);
    cudaMalloc(&b, sizeof(int)*1000);

    getErrorCuda((copyKernel<<<100, 128>>>(a, b, 2, 1000)));
    cudaDeviceSynchronize();
}