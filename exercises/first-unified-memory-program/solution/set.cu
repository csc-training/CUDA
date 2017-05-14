// Note that in this model we do not check
// the error codes and status of kernel call.

#include <cstdio>
#include <cmath>

__global__ void set(int *A, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    A[idx] = idx;
}


int main(void)
{
    const int N = 128;
    int *A;
    
    cudaMallocManaged((void**)&A, N * sizeof(int));

    set<<<2, 64>>>(A, N);
    cudaDeviceSynchronize();
    

    for (int i = 0; i < N; i++)
      printf("%i ", A[i]);
    printf("\n");

    cudaFree((void*)A);
    return 0;
}
