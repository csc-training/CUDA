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


// Routine for stream test
void streamtest(double *hC, const double *hA, const double *hB,
                double *dC, const double *dA, const double *dB,
                stream *s, int nstreams, float *gputime, int tib,
                int iterations)
{
    // Add here the needed timing event calls
    cudaEvent_t start, stop;

    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    CUDA_CHECK( cudaEventRecord(start) );    

    for (int i = 0; i < nstreams; ++i) {
        // Add here the copy - kernel execution - copy sequence
        // for each stream
        //
        // Each stream will copy their own part of the input data
        // to the GPU starting from address &(dA[sidx]).
        // Size of the block is slen (see variables below).
        int sidx = s[i].start;
        int slen = s[i].len;
                                    
        #error Add here the asynchronous memory copies

        // You can use these values for the grid and block sizes
        dim3 grid, threads;
        grid.x = (slen + tib - 1) / tib;
        threads.x = tib;
       
        // Lauch the kernel to the correct stream. See the default stream
        // version for help. Use the stream s[i].strm. The data vector
        // addresses are &(dC[sidx]), etc.
        #error Add kernel launch to the stream

        // Adde the missing memory copy
        #error Add the asynchronous memory copy
    }    

    // Add the calls needed for execution timing and compute
    // the elapsed time to the gputime variable
    CUDA_CHECK( cudaEventRecord(stop) );
    CUDA_CHECK( cudaEventSynchronize(stop) );

    CHECK_ERROR_MSG("Stream test failed");

    CUDA_CHECK( cudaEventElapsedTime(gputime, start, stop) );

    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
}

// Routine for default stream reference
void default_stream(double *hC, const double *hA, const double *hB,
                    double *dC, const double *dA, const double *dB,
                    int N, float *gputime, int tib, int iterations)
{
    // Add here the needed timing event calls
    cudaEvent_t start, stop;

    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    CUDA_CHECK( cudaEventRecord(start) );

    // Non-asynchronous copies
    CUDA_CHECK( cudaMemcpy((void *)dA, (void *)hA,
                            sizeof(double) * N,
                            cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaMemcpy((void *)dB, (void *)hB,
                            sizeof(double) * N,
                            cudaMemcpyHostToDevice) );

    dim3 grid, threads;
    grid.x = (N + tib - 1) / tib;
    threads.x = tib;

    vector_add<<<grid, threads>>>(dC, dA, dB, N, iterations);

    CUDA_CHECK( cudaMemcpyAsync((void *)hC, (void *)dC,
				sizeof(double) * N,
				cudaMemcpyDeviceToHost) );

    // Add the calls needed for execution timing and compute
    // the elapsed time to the gputime variable
    CUDA_CHECK( cudaEventRecord(stop) );
    CUDA_CHECK( cudaEventSynchronize(stop) );

    CHECK_ERROR_MSG("Default stream test failed");

    CUDA_CHECK( cudaEventElapsedTime(gputime, start, stop) );

    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
}

// Create the streams and compute the decomposition
void create_streams(int nstreams, int vecsize, stream **strm)
{
    *strm = new stream[nstreams];
    stream *s = *strm;
    for(int i = 0; i < nstreams; i++) {
        CUDA_CHECK( cudaStreamCreate(&s[i].strm) );
    }

    s[0].start = 0;
    s[0].len = vecsize / nstreams;
    s[0].len += vecsize % nstreams ? 1 : 0;
    for(int i = 1; i < nstreams; i++) {
        int offset = vecsize / nstreams;
        if(i < vecsize % nstreams) {
            offset++;
        }
        s[i].len = offset;
        s[i].start = s[i-1].start + offset;
    }
}

// Delete the streams
void destroy_streams(int nstreams, stream *s)
{
    for(int i = 0; i < nstreams; i++) {
        CUDA_CHECK( cudaStreamDestroy(s[i].strm) );
    }
    delete[] s;
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
    // of streams.
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

