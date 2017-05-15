#include <cstdio>
#include <cassert>

const int blocksize = 96;

// kernel prototypes
__global__ void k1(void);
__global__ void k2(void);

// array of values to fill
__device__ int data[blocksize];

// kernel that fills the array in device memory
__global__ void k2(void)
{
    int idx = threadIdx.x;
    if (idx < blocksize) {
        data[idx] = idx;
    }
}

// kernel that calls the fill kernel
__global__ void k1(void)
{
    int idx = threadIdx.x;

    if (idx == 0) {
        k2<<<1, blocksize>>>();
    }

    printf("Thread %i has value %i\n", idx, data[idx]);
}

int main(void)
{
    k1<<<1, blocksize>>>();
    cudaDeviceSynchronize();
    return 0;
}
