#include <cstdio>
#include <cmath>
#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG


/* Information of the decomposition */
struct Decomp {
    int len;
    int start;
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

    // Check that we have two CUDA devices available    
    CUDA_CHECK( cudaGetDeviceCount(&devicecount) );
    switch (devicecount) {
    case 0:
        printf("Could not find any CUDA devices!\n");
        exit(EXIT_FAILURE);
    case 1:
        printf("Found one CUDA device, this program requires two\n");
        exit(EXIT_FAILURE);
    default:
        printf("Found (at least) two CUDA devices.\n");
    }


    if (argc < 2) {
        printf("Usage: %s N\nwhere N is the length of the vector.\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    CUDA_CHECK( cudaSetDevice(0) );
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    CUDA_CHECK( cudaMallocHost((void**)&hA, sizeof(double) * N) );
    CUDA_CHECK( cudaMallocHost((void**)&hB, sizeof(double) * N) );
    CUDA_CHECK( cudaMallocHost((void**)&hC, sizeof(double) * N) );

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
        CUDA_CHECK( cudaSetDevice(i) );
        CUDA_CHECK( cudaMalloc((void**)&dA[i], sizeof(double) * dec[i].len) );
        CUDA_CHECK( cudaMalloc((void**)&dB[i], sizeof(double) * dec[i].len) );
        CUDA_CHECK( cudaMalloc((void**)&dC[i], sizeof(double) * dec[i].len) );
        CUDA_CHECK( cudaStreamCreate(&(strm[i])) );
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
        CUDA_CHECK( cudaSetDevice(i) );

        CUDA_CHECK( cudaMemcpyAsync(dA[i], (void *)&(hA[dec[i].start]), 
                                    sizeof(double) * dec[i].len, 
                                    cudaMemcpyHostToDevice, strm[i]) );
        
        CUDA_CHECK( cudaMemcpyAsync(dB[i], (void *)&(hB[dec[i].start]),
                                    sizeof(double) * dec[i].len, 
                                    cudaMemcpyHostToDevice, strm[i]) );
        
        dim3 grid, threads;
        grid.x = (dec[i].len + ThreadsInBlock - 1) / ThreadsInBlock;
        threads.x = ThreadsInBlock;
        
        vector_add<<<grid, threads, 0, strm[i]>>>(dC[i], dA[i], dB[i], 
                                                  dec[i].len);
        
        CUDA_CHECK( cudaMemcpyAsync((void *)&(hC[dec[i].start]), dC[i],
                                    sizeof(double) * dec[i].len, 
                                    cudaMemcpyDeviceToHost, strm[i]) );
    }

    //// Add here the stream synchronization calls. After both
    // streams have finished, we know that we stop the timing.
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK( cudaSetDevice(i) );
        CUDA_CHECK( cudaStreamSynchronize(strm[i]) );
        CUDA_CHECK( cudaStreamDestroy(strm[i]) );
    }

    // Add here the timing event calls
    CUDA_CHECK( cudaSetDevice(0) );
    CUDA_CHECK( cudaEventRecord(stop) );

    /* Release device memories */
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK( cudaSetDevice(i) );
        CUDA_CHECK( cudaFree((void*)dA[i]) );
        CUDA_CHECK( cudaFree((void*)dB[i]) );
        CUDA_CHECK( cudaFree((void*)dC[i]) );
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
