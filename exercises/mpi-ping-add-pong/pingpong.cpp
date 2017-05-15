#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "error_checks.h"

// Kernel call wrapper function prototype 
void call_kernel(double *data, int N, int blocksize, int tib);


/* 
   This routine can be used to inspect the properties of a node
   Return arguments:
   
   nodeName (char *) -- Name of the node as returned by MPI_Get_processor_name
   nodeRank (int *)  -- My rank in the node communicator
   nodeProcs (int *) -- Total number of processes in this node
   devCount (int *) -- Number of CUDA devices available in the node
*/
void getNodeInfo(char *nodeName, int *nodeRank, int *nodeProcs, int *devCount)
{
    unsigned hostid;
    int nodenamelen;
    MPI_Comm intranodecomm;
    
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,  MPI_INFO_NULL, &intranodecomm);

    MPI_Comm_rank(intranodecomm, nodeRank);
    MPI_Comm_size(intranodecomm, nodeProcs);

    MPI_Comm_free(&intranodecomm);
    CUDA_CHECK( cudaGetDeviceCount(devCount) );
}


/* Test routine for CPU-to-CPU copy */
void CPUtoCPUtest(int rank, double *data, int N, double &timer)
{
    double start, stop;
    
    start = MPI_Wtime();
    
    if (rank == 0) {
        MPI_Send(data, N, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD);
        MPI_Recv(data, N, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(data, N, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Add one*/
        for (int i = 0; i < N; ++i)
            data[i] += 1.0;
        
        MPI_Send(data, N, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }

    stop = MPI_Wtime();

    timer = stop - start;
}


/* Test routine for GPU-CPU-to-CPU-GPU copy */
void GPUtoGPUtestManual(int rank, double *hA, double *dA, int N, double &timer)
{
   //TODO: Implement transfer here that uses manual copies to host, and MPI on
   //host. Remember to add one as in CPU code 
 
}


/* Test routine for GPU-CPU-to-CPU-GPU copy */
void GPUtoGPUtestCudaAware(int rank, double *dA, int N, double &timer)
{
   //TODO: Implement transfer here that uses CUDA-aware MPI to transfer data
}



/* Simple ping-pong main program */
int main(int argc, char *argv[])
{
    char nodename[MPI_MAX_PROCESSOR_NAME];
    int rank, nprocs, noderank, nodenprocs, devcount;

    double *dA, *hA;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        printf("Need the vector length as argument\n");
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    getNodeInfo(nodename, &noderank, &nodenprocs, &devcount);

    /* Due to the test, we need exactly two processes with one GPU for
       each */
    if (nprocs != 2) {
        printf("Need exactly two processes!\n");
        exit(EXIT_FAILURE);
    }
    if (devcount == 0) {
        printf("Could now find any CUDA devices.\n");
        exit(EXIT_FAILURE);
    }
    if (nodenprocs > devcount) {
        printf("Not enough GPUs for all processes in the node.\n");
        exit(EXIT_FAILURE);
    }

    //TODO: Select the device according to the node rank
    //TODO: allocate device memories

    /* Initialize and copy the data to the device memory */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice) );
    
    /* CPU-to-CPU test */
    double CPUtime;
    CPUtoCPUtest(rank, hA, N, CPUtime);

    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        
        printf("CPU-CPU time %f, errorsum %f\n", CPUtime, errorsum);
    }

    /* Initialize and copy the data to the device memory */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice) );


    /* GPU-to-GPU test */
    double GPUtime;
    GPUtoGPUtest(rank, hA, dA, N, GPUtime);

    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        
        printf("GPU-GPU manual time %f, errorsum %f\n", GPUtime, errorsum);
    }

    /* Initialize and copy the data to the device memory */
    for (int i = 0; i < N; ++i)
       hA[i] = 1.0;    
    CUDA_CHECK( cudaMemcpy(dA, hA, sizeof(double)*N, cudaMemcpyHostToDevice) );

    /* GPU-to-GPU test */
    double GPUtime;
    GPUtoGPUtest(rank, hA, dA, N, GPUtime);

    if (rank == 0) {
        double errorsum = 0;
        for (int i = 0; i < N; ++i)
            errorsum += hA[i] - 2.0;
        
        printf("GPU-GPU cuda-aware time %f, errorsum %f\n", GPUtime, errorsum);
    }

    MPI_Finalize();
    return 0;
}
