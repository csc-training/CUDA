// This is a modified version of code presented in nvidia blog:
// https://devblogs.nvidia.com/parallelforall/introduction-cuda-dynamic-parallelism
// Original code by Andrew V. Adinetz, licensed under the MIT license is available
// at github: https://github.com/canonizer/mandelbrot-dyn
//
// See LICENSE for full license information

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <thrust/complex.h>
#include "error_checks.h"

extern "C" {
#include "pngwriter.h"
}

// Maximum number of iterations
const int MAX_ITER_COUNT = 512;
// Value for a neutral pixel
const int NEUTRAL_PIXEL = MAX_ITER_COUNT + 1;
// Marker for different iteration counts
const int DIFF_ITER_COUNT = -1;
// Block size along X and Y axes
const int BSX = 64;
const int BSY = 4;
// Maximum recursion depth
const int MAX_DEPTH = 8;
// Region size below which do per-pixel
const int MIN_SIZE = 32;
// Subdivision factor along each axis
const int SUBDIV = 2;
// Initial subdivision when launched from host
const int INIT_SUBDIV = 1;

// Use the complex number type from thrust
typedef thrust::complex<float> complex;

// Time spent in device
double gpu_time = 0;

// Helper function, divides x by y and rounds up to the next integer
__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

// |z|^2 of a complex number z
__host__ __device__ float abs2(complex v)
{
    return v.real() * v.real() + v.imag() * v.imag();
}

// Find the iteration count for the pixel
__device__ int pixel_iterations(int w, int h, complex cmin, complex cmax,
        int x, int y)
{
    complex dc = cmax - cmin;
    float fx = (float)x / w;
    float fy = (float)y / h;
    complex c = cmin + complex(fx * dc.real(), fy * dc.imag());
    int iteration = 0;
    complex z = c;
    while(iteration < MAX_ITER_COUNT && abs2(z) < 2 * 2) {
        z = z * z + c;
        iteration++;
    }
    return iteration;
} 

// Binary operation for comparing two iteration counts
// If values are the same, the common value is returned.
// If one of the values is "neutral", the non-neutral value 
// is returned.
__device__ int compare_counts(int d1, int d2)
{
    if(d1 == d2)
        return d1;
    else if(d1 == NEUTRAL_PIXEL || d2 == NEUTRAL_PIXEL)
        return min(d1, d2);
    else
        return DIFF_ITER_COUNT;
}

// Evaluates the common iteration count on the border, if it exists
__device__ int border_iterations(int w, int h, complex cmin, complex cmax,
        int x0, int y0, int d)
{
    // Check whether all boundary pixels have the same iteration count 
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bs = blockDim.x * blockDim.y;
    int comm_iter_count = NEUTRAL_PIXEL;
    // For all boundary pixels, distributed across threads
    for(int r = tid; r < d; r += bs) {
        // For each boundary: b = 0 is east, then counter-clockwise
        for(int b = 0; b < 4; b++) {
            int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int iters = pixel_iterations(w, h, cmin, cmax, x, y);
            comm_iter_count = compare_counts(comm_iter_count, iters);
        }
    }

    // Reduce across threads in the block
    __shared__ int loc_iters[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if(tid < nt)
        loc_iters[tid] = comm_iter_count;
    __syncthreads();
    for(; nt > 1; nt /= 2) {
        if(tid < nt / 2)
            loc_iters[tid] = compare_counts(loc_iters[tid], loc_iters[tid + nt / 2]);
        __syncthreads();
    }
    return loc_iters[0];
}

// The kernel to fill the image region with a specific dwell value
__global__ void iter_fill_k(int *iters, int w, int x0, int y0, int d, int iter_count)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < d && y < d) {
        x += x0, y += y0;
        iters[y * w + x] = iter_count;
    }
} 

// The kernel to count per-pixel values of the portion of the Mandelbrot set
__global__ void mandelbrot_pixel_k(int *iter_counts, int w, int h, complex cmin,
        complex cmax, int x0, int y0, int d)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x < d && y < d) {
        x += x0, y += y0;
        iter_counts[y * w + x] = pixel_iterations(w, h, cmin, cmax, x, y);
    }
}

/* Computes the iteration counts for Mandelbrot image using dynamic parallelism;
        one block is launched per pixel
        @param iter_counts the output array
        @param w the width of the output image
        @param h the height of the output image
        @param cmin the complex value associated with the left-bottom corner of the
        image
        @param cmax the complex value associated with the right-top corner of the
        image
        @param x0 the starting x coordinate of the portion to compute
        @param y0 the starting y coordinate of the portion to compute
        @param d the size of the portion to compute (the portion is always a square)
        @param depth kernel invocation depth
        @remarks the algorithm reverts to per-pixel Mandelbrot evaluation once
        either maximum depth or minimum size is reached
 */
__global__ void mandelbrot_block_k(int *iter_counts, int w, int h, complex cmin,
        complex cmax, int x0, int y0, int d, int depth)
{
    // Origin of this block
    x0 += d * blockIdx.x;
    y0 += d * blockIdx.y;

    // Check if the block boundary has common iteration count
    int comm_iter_count = border_iterations(w, h, cmin, cmax, x0, y0, d);

    // Only single thread in a block will call new kernels
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if(comm_iter_count != DIFF_ITER_COUNT) {
            // Uniform block, fill with common value
            dim3 bs(BSX, BSY);
            dim3 grid(divup(d, BSX), divup(d, BSY));
            iter_fill_k<<<grid, bs>>>(iter_counts, w, x0, y0, d, comm_iter_count);
        } else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // Subdivide recursively
            dim3 bs(BSX, BSY);
            dim3 grid(SUBDIV, SUBDIV);
            mandelbrot_block_k<<<grid, bs>>>(iter_counts, w, h, cmin, cmax,
                    x0, y0, d / SUBDIV, depth + 1);
        } else {
            // Last level, fill pixel-by-pixel
            dim3 bs(BSX, BSY);
            dim3 grid(divup(d, BSX), divup(d, BSY));
            mandelbrot_pixel_k<<<grid, bs>>>(iter_counts, w, h, cmin, cmax,
                    x0, y0, d);
        }
        CHECK_ERROR_MSG("mandelbrot_block_k");
    }
}


int main(int argc, char **argv)
{
    cudaEvent_t start, stop, copystop;
    float gputime, copytime;

    // Picture size, should be power of two
    const int w = 16384;
    const int h = w;
    int *h_iter_counts, *d_iter_counts;
    
    int pic_bytes = w * h * sizeof(int);

    CUDA_CHECK( cudaMalloc((void**)&d_iter_counts, pic_bytes) );
    h_iter_counts = (int*)malloc(pic_bytes);

    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );
    CUDA_CHECK( cudaEventCreate(&copystop) );

    CUDA_CHECK( cudaEventRecord(start) );

    dim3 bs(BSX, BSY);
    dim3 grid(INIT_SUBDIV, INIT_SUBDIV);

    // Launch the recursive kernel with initial number of blocks
    mandelbrot_block_k<<<grid, bs>>>(d_iter_counts, w, h, complex(-1.5, -1),
            complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1);
 
    // Synchronize for timing   
    CUDA_CHECK( cudaThreadSynchronize() );
    CUDA_CHECK( cudaEventRecord(stop) );

    CUDA_CHECK( cudaMemcpy(h_iter_counts, d_iter_counts, pic_bytes, 
                    cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaEventRecord(copystop) );

    CUDA_CHECK( cudaEventSynchronize(copystop) );
    CUDA_CHECK( cudaEventElapsedTime(&gputime, start, stop) );
    CUDA_CHECK( cudaEventElapsedTime(&copytime, stop, copystop) );
    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
    CUDA_CHECK( cudaEventDestroy(copystop) );

    // Save the image to a PNG file
#ifndef SKIP_PNG_WRITING
    save_png(h_iter_counts, w, h, "mandelbrot.png");
#endif
    // Print the timings
    printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n",
                gputime / 1000., h * w * 1e-6 * 1000. / gputime );
    printf("Copying took %.3lf s\n", copytime / 1000.);

    cudaFree(d_iter_counts);
    free(h_iter_counts);
    return 0;
}

