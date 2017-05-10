// Note that in this model we do not check
// the error codes and status of kernel call.

#include <cstdio>
#include <cmath>

__global__ void set(int *A, int N)
{
  // TODO 3 - Complete kernel code
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    A[idx] = idx;
}


int main(void)
{
    const int N = 128;
    
    int *d_A;
    int *h_A;

    h_A = (int*) malloc(N * sizeof(int));
    
    // TODO 1 - Allocate memory for device pointer d_A
    cudaMalloc((void**)&d_A, N * sizeof(int));

    // TODO 4 - Call kernel set()
    set<<<2, 64>>>(d_A, N);

    // TODO 5 - Copy the results from device memory
    cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
      printf("%i ", h_A[i]);
    printf("\n");

    free(h_A);
    
    // TODO 2 - Free memory for device pointer d_A
    cudaFree((void*)d_A);

    return 0;
}
