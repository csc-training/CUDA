#include <cstdio>
#include <cmath>
#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG

// Number of iterations in the kernel
#define ITER_MULTIPLIER 4

// Information of stream for simple domain decomposition
struct stream {
    cudaStream_t strm;   // Stream
    int len;             // Length of the part for this stream
    double *A, *B, *C;
};

// Kernel for vector summation
__global__ void vector_add(double *C, const double *A, const double *B,
                           int N, int iterations){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Do not try to access past the allocated memory
    for (int i = idx; i < N; i += stride) {
        C[i] = 0;
        for (int j = 0; j < ITER_MULTIPLIER * iterations; j++) {
            C[i] += A[i] + B[i];
        }
    }
}


// Routine for stream test
void streamtest(stream *s, int nstreams,  int tib,
                int iterations)
{

    for (int i = 0; i < nstreams; ++i) {
        // Add here the copy - kernel execution - copy sequence
        // for each stream
        int slen = s[i].len;

        dim3 grid, threads;
        grid.x = (slen + tib - 1) / tib;
        threads.x = tib;
        
        //set value on CPU
        for(int j = 0; j < s[i].len; ++j) {
           s[i].A[j] = 1.0;
           s[i].B[j] = 2.0;
        }

        vector_add<<<grid, threads, 0, s[i].strm>>>(s[i].C, s[i].A, s[i].B, slen, iterations);
    }    

    for (int i = 0; i < nstreams; ++i) {
       //check value
        double errorsum = 0;
        const double ref_value = 3.0 * ITER_MULTIPLIER * iterations;

        //TODO Here the CPU accesses C array, make sure it is allowed to. 
        for(int j = 0; j < s[i].len; ++j) {
            errorsum += s[i].C[j] - ref_value;              
        }
        printf("Errorsum is %g on stream %d\n", errorsum, i);
    }
}

// Create the streams and compute the decomposition
void create_streams(int nstreams, int vecsize, stream **strm)
{
    *strm = new stream[nstreams];
    stream *s = *strm;
    for(int i = 0; i < nstreams; i++) {
        CUDA_CHECK( cudaStreamCreate(&s[i].strm) );
    }

    s[0].len = vecsize / nstreams;
    s[0].len += vecsize % nstreams ? 1 : 0;
    

    for(int i = 1; i < nstreams; i++) {
       int add = vecsize / nstreams;
       if(i < vecsize % nstreams) {
          add++;
       }
       s[i].len = add;
    }
    
    for(int i = 0; i < nstreams; i++) {
       //TODO: Add here allocations for managed memory

       //TODO: Attach them to streams to enable independent operation of the various streams

    }
    
}

// Delete the streams
void destroy_streams(int nstreams, stream *s)
{
    for(int i = 0; i < nstreams; i++) {
        CUDA_CHECK( cudaStreamDestroy(s[i].strm) );
        //TODO: Free memory allocations


    }
    delete[] s;
}



int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 512;
    int iterations;
    stream *s;

    cudaDeviceProp prop;

    if (argc < 2) {
        printf("Usage: %s N\nwhere N is the length of the vector.\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    // Determine the number of available multiprocessors on the device.
    // It is used for a coarse adjustment of the computation part of
    // this test.
    cudaGetDeviceProperties(&prop, 0);
    iterations = (prop.multiProcessorCount + 1) / 2;


    // Now do the addition with streams, note that each stream will need to allocate its
    // own memory area 
    int stream_count = 8;
    create_streams(stream_count, N, &s);
    streamtest(s, stream_count,  ThreadsInBlock, iterations);
    destroy_streams(stream_count, s);


    return 0;
}

