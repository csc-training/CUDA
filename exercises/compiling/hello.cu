// Note that in this model we do not check
// the error codes and status of kernel call.

#include <cstdio>
#include <cmath>

__global__ void hello()
{
  printf("Greetings from your GPU\n");
}


int main(void)
{
  int count, device;
  cudaGetDeviceCount(&count);
  cudaGetDevice(&device);
  printf("You have in total %d GPUs in your system\n", count);
  printf("GPU %d will now print a message for you:\n", device);

  hello<<<1,1>>>();
  cudaDeviceSynchronize();
  
  return 0;	
}
