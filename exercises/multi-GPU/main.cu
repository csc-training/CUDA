#include <cstdio>
#include <cmath>
#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG


/* Information of the decomposition */
struct Decomp {
    int len; // the lenght of the array for the current device
    int start; // the start index for the array on the current device
};


/* Kernel for vector summation */
__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 128;
    double *dA[2], *dB[2], *dC[2];
    double *hA, *hB, *hC;
    int devicecount;
    cudaEvent_t start, stop;
    cudaStream_t strm[2];
    Decomp dec[2];

    #error Check that we have two CUDA devices available    

    if (argc < 2) {
        printf("Usage: %s N\nwhere N is the length of the vector.\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    #error Select device 0 and create timing events and allocate pinned host memory

    // Here we initialize the host memory values    
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    /* The decomposition */
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    /* Allocate memory for the devices and per device streams */
    for (int i = 0; i < 2; ++i) {
        #error Allocate memory areas and streams for each device
    }

    /* Start timer */
    CUDA_CHECK( cudaSetDevice(0) );
    CUDA_CHECK( cudaEventRecord(start) );

    /* Copy the parts of the vectors on host to the devices and
       execute a kernel for each part. Note that we use asynchronous
       copies and streams. Without this the execution is serialized
       because the memory copies block the host process execution. */
    for (int i = 0; i < 2; ++i) {
        // Start by selecting the active device!
        #error Add here the memcpy-kernel-memcpy parts
    }

    //// Add here the stream synchronization calls. After both
    // streams have finished, we know that we stop the timing.
    for (int i = 0; i < 2; ++i) {
        #error Add here the synchronization calls
    }

    // Add here the timing event calls
    #error Add here timing calls

    /* Release device memories */
    for (int i = 0; i < 2; ++i) {
        #error Add here the cleanup code
    }

    int errorsum = 0;

    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }

    printf("Error sum = %i\n", errorsum);

    // Compute the elapsed time and release host memory
    float gputime;
    CUDA_CHECK( cudaSetDevice(0) );
    CUDA_CHECK( cudaEventElapsedTime(&gputime, start, stop) );
    printf("Time elapsed: %f\n", gputime / 1000.);

    CUDA_CHECK( cudaFreeHost((void*)hA) );
    CUDA_CHECK( cudaFreeHost((void*)hB) );
    CUDA_CHECK( cudaFreeHost((void*)hC) );

    return 0;
}
