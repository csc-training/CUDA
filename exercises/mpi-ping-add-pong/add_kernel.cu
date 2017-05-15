
/* Very simple addition kernel */
__global__ void add_kernel(double *in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        in[tid]++;
}


/* 
   Kernel call wrapper, we can not use the <<<>>> syntax in MPI code.
   Arguments: 
   data (double *) -- pointer to the device memory data
   N         (int) -- number of elements in data vector
   blocksize (int) -- number of blocks in kernel call
   tib       (int) -- number of threads in a block in kernel call
*/
void call_kernel(double *data, int N, int blocksize, int tib)
{
    add_kernel<<<blocksize, tib>>> (data, N);
}
