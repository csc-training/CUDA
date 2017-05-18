#include <iostream>
#include <cuda_runtime_api.h>
#include <chrono>

#define DEBUG_DEV

#ifdef DEBUG_DEV
#define getErrorCuda(command)\
		command;\
		cudaDeviceSynchronize();\
		cudaThreadSynchronize();\
		if (cudaPeekAtLastError() != cudaSuccess){\
			std::cout << #command << " : " << cudaGetErrorString(cudaGetLastError())\
			 << " in file " << __FILE__ << " at line " << __LINE__ << std::endl;\
			exit(1);\
		}
#endif
#ifndef DEBUG_DEV
#define getErrorCuda(command) command;
#endif

__constant__ float const_stencilWeight[21];


// base case
__global__ void stencil(float *src, float *dst, int size, float *stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += src[idx+i] * stencilWeight[i+10];
    }
    dst[idx] = out;
}

// read only cache stencil coefficients
__global__ void stencilReadOnly1(float *src, float *dst, int size, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += src[idx+i] * __ldg(&stencilWeight[i+10]);
    }
    dst[idx] = out;
}

// read only data
__global__ void stencilReadOnly2(float *src, float *dst, int size, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += __ldg(&src[idx+i]) * stencilWeight[i+10];
    }
    dst[idx] = out;
}

// read only coefficients and data
__global__ void stencilReadOnly3(float *src, float *dst, int size, float* stencilWeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += __ldg(&src[idx+i]) * __ldg(&stencilWeight[i+10]);
    }
    dst[idx] = out;
}

// constat memory coefficients
__global__ void stencilConst1(float *src, float *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += src[idx+i] * const_stencilWeight[i+10];
    }
    dst[idx] = out;
}

// constant memory coefficients and data through read only cache
__global__ void stencilConst2(float *src, float *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx += 11;
    if (idx >= size)
        return;
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += __ldg(&src[idx+i]) * const_stencilWeight[i+10];
    }
    dst[idx] = out;
}

// constant memory coefficients and data from shared 
__global__ void stencilShared1(float *src, float *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[1024+21];
    for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
    {
        buffer[i] = src[idx+i];
    }
    idx += 11;
    if (idx >= size)
        return;
    
    __syncthreads();
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += buffer[threadIdx.x+10+i] * const_stencilWeight[i+10];
    }
    dst[idx] = out;
}

// constant memory coefficients and data from shared thorugh read only
__global__ void stencilShared2(float *src, float *dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer[1024+21];
    for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
    {
        buffer[i] = __ldg(&src[idx+i]);
    }
    idx += 11;
    if (idx >= size)
        return;
    
    __syncthreads();
    float out = 0;
    #pragma unroll
    for(int i = -10;i < 10; i++)
    {
        out += buffer[threadIdx.x+10+i] * const_stencilWeight[i+10];
    }
    dst[idx] = out;
}

bool verify(float *arr, float *corr, int count)
{
    // skip the first elements since they may be wrong
    for(int i = 11; i < count; i++)
    {
        if(arr[i] != corr[i])
        {   
            std::cout << "check failed" << i << " " << arr[i] << " != " << corr[i] << std::endl;
            exit(1);
        }
    }
}

int main()
{
    float *a;
    float *b;
    float *bOut;
    float *bCorr;
    float *weights;
    getErrorCuda(cudaMalloc(&a, sizeof(float)*102400000));
    getErrorCuda(cudaMalloc(&b, sizeof(float)*102400000));
    getErrorCuda(cudaMallocHost(&bOut, sizeof(float)*102400000));
    getErrorCuda(cudaMallocManaged(&bCorr, sizeof(float)*102400000));
    getErrorCuda(cudaMallocManaged(&weights, sizeof(float)*21));

    cudaDeviceSynchronize();    

    for(int i = 0; i < 102400000;i++)
    {
        //a[i] = 0;
        //b[i] = 0;
        bCorr[i] = 0;
    }

    cudaMemset(a, 1, 102400000);
    cudaMemset(b, 1, 102400000);
    cudaMemset(bCorr, 1, 102400000);
    cudaMemset(bOut, 1, 102400000);

    cudaDeviceSynchronize();    
    
    int blockSize = 1024;
    int blocks = 10000;
    for(int i = 0; i < 21;i++)
        weights[i] = i-10;
    
    
    cudaDeviceSynchronize();    


    // copy to constant memory    
    cudaMemcpyToSymbol(const_stencilWeight, weights, sizeof(float)*21);

    // run the basic case once to get the "correct" results
    getErrorCuda((stencil<<<blocks, blockSize>>>(a, bCorr, 10240000, weights)));
    cudaDeviceSynchronize();    

    getErrorCuda((stencil<<<blocks, blockSize>>>(a, b, 10240000, weights)));
    cudaDeviceSynchronize(); 
    getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
    verify(bOut, bCorr, 1000);

    cudaSetDevice(0); 


    float minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencil<<<blocks, blockSize>>>(a, b, 10240000, weights)));
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  

	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "Non optimized " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    std::cout << std::endl;

    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilReadOnly1<<<blocks, blockSize>>>(a, b, 10240000, weights)));
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only cache stencil coefficients " <<(blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilReadOnly2<<<blocks, blockSize>>>(a, b, 10240000, weights)));
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only data " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilReadOnly3<<<blocks, blockSize>>>(a, b, 10240000, weights)));
        cudaDeviceSynchronize();  
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
        
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "read only coefficients and data " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;

    std::cout << std::endl;

        for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilConst1<<<blocks, blockSize>>>(a, b, 10240000)));
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();

        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients " << (blockSize*blocks)/minTime << " updates/s" << std::endl;

    minTime = 10000;


        for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilConst2<<<blocks, blockSize>>>(a, b, 10240000)));
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();

        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data through read only cache " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    std::cout << std::endl;


    minTime = 10000;
            for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilShared1<<<blocks, blockSize>>>(a, b, 10240000)));
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data from shared " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;
    minTime = 10000;
    for(int i  = 0; i < 10; i++)
    {
        cudaDeviceSynchronize();  
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
	    start = std::chrono::system_clock::now();
        getErrorCuda((stencilShared2<<<blocks, blockSize>>>(a, b, 10240000)));
        cudaDeviceSynchronize();    
        end = std::chrono::system_clock::now();
        
        getErrorCuda(cudaMemcpy(bOut, b, sizeof(float)*10240000, cudaMemcpyDefault));
        verify(bOut, bCorr, 1000);  
	    std::chrono::duration<float> elapsed_seconds = end-start;
        minTime = std::min(elapsed_seconds.count(), minTime);
    }
    std::cout << "constant memory coefficients and data from shared thorugh read only " << (blockSize*blocks)/minTime << " updates/s" << std::endl;
    minTime = 10000;


}
