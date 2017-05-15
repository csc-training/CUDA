#ifndef EX3_H_
#define EX3_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

// Helper function prototypes
double compareArrays(const double *a, const double *b, int N);
double diffCPU(const double *a, const double *b, int N);
void sweepCPU(double *phi, const double *phiPrev, 
              const double *source, double h2, int N);


/* -------------------------------------------------------------------------
   EXTRACURRICULAR ACTIVITIES 
   
   This part provides the reduction operation (in this case summation of
   difference of two arrays) using thrust library. Thrust mimics the
   syntax and design of standard template library (STL) of C++. Thrust is
   also a part of CUDA 4 SDK.

   More information can be found from thrust home page 
   http://code.google.com/p/thrust/
   ----------------------------------------------------------------------- */

template<typename T>
class square_diff_thr : public thrust::unary_function<thrust::tuple<T, T>, T>
{
public:
    __host__ __device__ 
    T operator()(const thrust::tuple<T, T>& x) const {
        return (thrust::get<1>(x) - thrust::get<0>(x)) * 
            (thrust::get<1>(x) - thrust::get<0>(x));
    }
};

template<typename T>
class square_thr : public thrust::unary_function<T, T>
{
public:
    __host__ __device__
    T operator()(const T& x) const {
        return x*x;
    }
};

template<typename T>
T diffGPU(T *A_d, T *B_d, int N)
{
    typedef thrust::device_ptr<T> FloatIterator;
    typedef thrust::tuple<FloatIterator, FloatIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
    
    thrust::device_ptr<T> A_ptr(A_d);
    thrust::device_ptr<T> B_ptr(B_d);
    
    ZipIterator first = 
        thrust::make_zip_iterator(thrust::make_tuple(A_ptr, B_ptr));
    ZipIterator last = 
        thrust::make_zip_iterator(thrust::make_tuple(A_ptr + N*N, 
                                                     B_ptr + N*N));
    
    T a1 = thrust::transform_reduce(first, last, square_diff_thr<T>(), 
                                  static_cast<T>(0), thrust::plus<T>());
    T a2 = thrust::transform_reduce(B_ptr, B_ptr + N*N, 
                                  square_thr<T>(), static_cast<T>(0),
                                  thrust::plus<T>());
    
    return sqrt(a1/a2);
}


#endif // EX3_H_
