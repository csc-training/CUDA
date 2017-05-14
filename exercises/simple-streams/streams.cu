#include <cstdio>
#include <cmath>
#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG

// Number of iterations in the kernel
#define ITER_MULTIPLIER 4
// Number of tests, number of streams doubles for each test. That is,
// there will be 2^N_TESTS streams in the last test
#define N_TESTS 4

// Information of stream for simple domain decomposition
struct stream {
    cudaStream_t strm;   // Stream
    int len;             // Length of the part for this stream
    int start;           // Offset to the start of the part
};

// Kernel for vector summation
__global__ void vector_add(double *C, const double *A, const double *B,
                           int N, int iterations)
{
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

/* Routine for the test a) in the exercise */
void streamtest_a(double *hC, const double *hA, const double *hB,
                  double *dC, const double *dA, const double *dB,
                  stream *s, float *gputime, int tib)
{
    // Add here the timing events
    cudaEvent_t start, stop;
    for (int i = 0; i < 2; ++i) {
        #error Add asynchronous copy-in calls for each stream (s[0] and s[1])
    }
    
    for (int i = 0; i < 2; ++i) {
        #error Add here the kernel calls
    }

    for (int i = 0; i < 2; ++i) {
        #error Copy the results back
    }    

    #error Add the event calls and compute the elapsed time to gputime variable
}

/* Routine for test b) in exercise */
void streamtest_b(double *hC, const double *hA, const double *hB,
                  double *dC, const double *dA, const double *dB,
                  stream *s, float *gputime, int tib)
{
    // Add here the needed timing event calls
    cudaEvent_t start, stop;

    #error Add the timing event call here

    for (int i = 0; i < 2; ++i) {
        #error Add here the copy - kernel execution - copy sequence for each stream
    }    

    //// Add the calls needed for execution timing and compute
    // the elapsed time to the gputime variable
    #error Add the timing calls and compute the elapsed time
}


int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 512;
    int iterations;
    double *dA, *dB, *dC;
    double *hA, *hB, *hC;
    double ref_value;
    float gputime_ref, gputimes[N_TESTS];
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

    #error Add here the host memory allocation routines (page-locked)
    
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    ref_value = 3.0 * ITER_MULTIPLIER;

    CUDA_CHECK( cudaMalloc((void**)&dA, sizeof(double) * N) );
    CUDA_CHECK( cudaMalloc((void**)&dB, sizeof(double) * N) );
    CUDA_CHECK( cudaMalloc((void**)&dC, sizeof(double) * N) );
    
    // Check the timings of default stream first
    default_stream(hC, hA, hB, dC, dA, dB, N, &gputime_ref, ThreadsInBlock,
                   iterations);

    // Here we loop over the test. On each iteration, we double the number
    // of streams
    for(int strm = 0; strm < N_TESTS; strm++) {
        int stream_count = 1<<strm;
        create_streams(stream_count, N, &s);
        streamtest(hC, hA, hB, dC, dA, dB, s, stream_count, &gputimes[strm],
                   ThreadsInBlock, iterations);
        destroy_streams(stream_count, s);
    }

    CUDA_CHECK( cudaFree((void*)dA) );
    CUDA_CHECK( cudaFree((void*)dB) );
    CUDA_CHECK( cudaFree((void*)dC) );

    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - ref_value;
    }

    printf("Error sum = %i\n", errorsum);
    printf("Time elapsed for reference run: %f\n", gputime_ref / 1000.);
    for(int i = 0; i < N_TESTS; i++) {
        printf("Time elapsed for test %2i:       %f\n", 1<<i,
                gputimes[i] / 1000.);
    }

    #error Add here the correct host memory freeing routines

    return 0;
}
