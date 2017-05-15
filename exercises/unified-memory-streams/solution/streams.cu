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
    double *A, *B, *C;
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
void streamtest(stream *s, int nstreams, float *gputime, int tib,
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
        int slen = s[i].len;

        dim3 grid, threads;
        grid.x = (slen + tib - 1) / tib;
        threads.x = tib;
        
        vector_add<<<grid, threads, 0, s[i].strm>>>(s[i].C, s[i].A, s[i].B,
                                                    slen, iterations);
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
void default_stream(double *C, const double *A, const double *B,
                    int N, float *gputime, int tib, int iterations)
{
    // Add here the needed timing event calls
    cudaEvent_t start, stop;

    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    CUDA_CHECK( cudaEventRecord(start) );

    dim3 grid, threads;
    grid.x = (N + tib - 1) / tib;
    threads.x = tib;

    vector_add<<<grid, threads>>>(C, A, B, N, iterations);

    CUDA_CHECK( cudaEventRecord(stop) );
    //EventSynchronize is sufficient to Synchronize managed memory(!)
    CUDA_CHECK( cudaEventSynchronize(stop) );

    CHECK_ERROR_MSG("Stream test c) failed");

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
    // Add here allocations for managed memory
       CUDA_CHECK( cudaMallocManaged((void**)&(s[i].A), sizeof(double) * s[i].len) );
       CUDA_CHECK( cudaMallocManaged((void**)&(s[i].B), sizeof(double) * s[i].len) );
       CUDA_CHECK( cudaMallocManaged((void**)&(s[i].C), sizeof(double) * s[i].len) );
//attach them to streams to enable independent operation of the various streams
       CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].A) );
       CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].B) );
       CUDA_CHECK( cudaStreamAttachMemAsync(s[i].strm, s[i].C) );
    }
    
}

// Delete the streams
void destroy_streams(int nstreams, stream *s)
{
    for(int i = 0; i < nstreams; i++) {
        CUDA_CHECK( cudaStreamDestroy(s[i].strm) );
        //free large memory allocations
        CUDA_CHECK( cudaFree((void*)s[i].A) );
        CUDA_CHECK( cudaFree((void*)s[i].B) );
        CUDA_CHECK( cudaFree((void*)s[i].C) );


    }
    delete[] s;
}



int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 512;
    int iterations;
    double *A, *B, *C;
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

    // Add here allocations for managed memory
    CUDA_CHECK( cudaMallocManaged((void**)&A, sizeof(double) * N) );
    CUDA_CHECK( cudaMallocManaged((void**)&B, sizeof(double) * N) );
    CUDA_CHECK( cudaMallocManaged((void**)&C, sizeof(double) * N) );
    
    for(int i = 0; i < N; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
    }


    
    // Check the timings of default stream first
    default_stream(C, A, B, N, &gputime_ref, ThreadsInBlock,
                   iterations);
    //free memory allocations
    CUDA_CHECK( cudaFree((void*)A) );
    CUDA_CHECK( cudaFree((void*)B) );
    CUDA_CHECK( cudaFree((void*)C) );


    int errorsum = 0;

    // Now do it with streams, note that each stream will need to allocate its
    // own memory area 
    // Here we loop over the test. On each iteration, we double the number
    // of streams
    for(int strm = 0; strm < N_TESTS; strm++) {
        int stream_count = 1<<strm;
        create_streams(stream_count, N, &s);

        //set value
        for(int i = 0; i < stream_count; ++i) {
           for(int j = 0; j < s[i].len; ++j) {
              s[i].A[j] = 1.0;
              s[i].B[j] = 2.0;
           }
        }
        

        streamtest(s, stream_count, &gputimes[strm], ThreadsInBlock, iterations);

        ref_value = 3.0 * ITER_MULTIPLIER * iterations;
        //check value
        for(int i = 0; i < stream_count; ++i) {
           for(int j = 0; j < s[i].len; ++j) {
              errorsum += s[i].C[j] - ref_value;              
           }
        }
        
        destroy_streams(stream_count, s);
    }


    printf("Error sum = %i\n", errorsum);
    printf("Time elapsed for reference run: %f\n", gputime_ref / 1000.);
    for(int i = 0; i < N_TESTS; i++) {
        printf("Time elapsed for test %2i:       %f\n", 1<<i,
               gputimes[i] / 1000.);
    }


    return 0;
}

