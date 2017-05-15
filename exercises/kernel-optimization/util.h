//
// Created by fredr on 22-04-2016.
//


#define DEBUG_DEV

#ifdef DEBUG_DEV
#define getErrorCuda(command)\
		command;\
		cudaDeviceSynchronize();\
		if (cudaPeekAtLastError() != cudaSuccess){\
			std::cout << #command << " : " << cudaGetErrorString(cudaGetLastError())\
			 << " in file " << __FILE__ << " at line " << __LINE__ << std::endl;\
			exit(1);\
		}
#endif
#ifndef DEBUG_DEV
#define getErrorCuda(command) command;
#endif

#include <cmath>
#include <math.h>


class Vec3 {
public:

    float x;
    float y;
    float z;
    __device__ __host__ Vec3(float x,float y,float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
	__device__ __host__ Vec3(float x)
	{
		this->x = x;
		this->y = x;
		this->z = x;
	}
    __device__ __host__ Vec3() = default;

    __device__ __host__ static inline float dot(const Vec3 &a, const Vec3 &b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __host__ static inline Vec3 cross(const Vec3 &a, const Vec3 &b)
    {
        return Vec3(a.y*b.z - a.z*b.y,
                    a.z*b.x - a.x*b.z,
                    a.x*b.y - a.y*b.x);
    }

    __device__ __host__ static inline Vec3 normalize(Vec3 v)
    {
        float invLen = 1/sqrtf(dot(v, v));
        return v * invLen;
    }
	__device__ __host__ static inline float length(const Vec3 v)
	{
		return sqrtf(dot(v, v));
	}
	__device__ __host__ inline Vec3 operator+(const Vec3 &b)
	{
		return Vec3(this->x + b.x, this->y + b.y, this->z + b.z);
	}
    __device__ __host__ inline Vec3 operator-(const Vec3 &b)
    {
        return Vec3(this->x - b.x, this->y - b.y, this->z - b.z);
    }
    __device__ __host__ inline Vec3 operator*(const Vec3 &b)
    {
        return Vec3(this->x * b.x, this->y * b.y, this->z * b.z);
    }
    __device__ __host__ inline  void operator*=(const Vec3 &b)
    {
        this->x *= b.x; this->y *= b.y; this->z *= b.z;
    }
    __device__ __host__ inline Vec3 operator*(const float &b)
    {
        return Vec3(this->x * b, this->y * b, this->z * b);
    }

};

__device__ __host__ inline void indToZorder(unsigned int &x, unsigned int &y, unsigned int ind)
{
    x = 0;
    y = 0;
    unsigned int mask = 1;
    while (ind != 0)
    {
        x |= ind & mask;
        ind = ind>>1;
        y |= ind & mask;
        mask = mask << 1;
    }
}

int writeImageRGB(char* filename, int width, int height, float *bufferR, float *bufferG, float *bufferB, char* title)
{
    int code = 0;
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;

    fp = fopen(filename, "wb");

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    info_ptr = png_create_info_struct(png_ptr);


    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    row = (png_bytep) malloc(3 * width * sizeof(png_byte));

    int x, y;
    for (y=0 ; y<height ; y++) {
        for (x=0 ; x<width ; x++) {            
            row[(x*3)+0] = bufferR[(y*width)+x] * 255;
            row[(x*3)+1] = bufferG[(y*width)+x] * 255;
            row[(x*3)+2] = bufferB[(y*width)+x] * 255;
        }
        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);

    return code;
}

float __device__ __host__ normalize(float max, float min, float val)
{
    return (val-min)/(max-min);
}

template <typename T>
__device__ __host__ T getMax(T a, T b)
{
    if(a>b)
        return a;
    return b;
}

template <typename T>
__device__ __host__ T getMin(T a, T b)
{
    if(a>b)
        return b;
    return a;
}

__device__ __host__ Vec3 clamp(Vec3 color) {
	return Vec3(getMax(getMin(1.f,color.x),0.f),
	            getMax(getMin(1.f,color.y),0.f),
	            getMax(getMin(1.f,color.z),0.f));
}
