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
    
    int *d_A;
    int *h_A;
    
    h_A = (int*) malloc(N * sizeof(int));
    
    cudaMalloc((void**)&d_A, N * sizeof(int));

    set<<<2, 64>>>(d_A, N);

    cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
      printf("%i ", h_A[i]);
    printf("\n");

    free(h_A);
    
    cudaFree((void*)d_A);
    return 0;
}
