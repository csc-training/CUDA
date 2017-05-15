
#include <iostream>
#include <png.h>
#include "util.h"
#include <limits>
#include <cuda_runtime_api.h>
#include <chrono>
#include <algorithm>

using namespace std;

__device__ __host__ float de(Vec3 pos) ;

__device__ __host__ Vec3 getNormal(Vec3 pos) 
{
	Vec3 xDir = Vec3(1,0,0);
	Vec3 yDir = Vec3(0,1,0);
	Vec3 zDir = Vec3(0,0,1);
	return Vec3::normalize(Vec3(de(pos+xDir)-de(pos-xDir),
	                            de(pos+yDir)-de(pos-yDir),
	                            de(pos+zDir)-de(pos-zDir)));
}


// distance "estimation" function 
__device__ __host__ float de(Vec3 pos) 
{
    float cutoff = 2;
    float power = 8;
	Vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < 50 ; i++) 
    {
		r = Vec3::length(z);
		if (r>cutoff) break;
		
		// convert to polar coordinates
		float theta = acosf(z.z/r);
		float phi = atanf(z.y);
		dr =  powf( r, power-1.0f)*power*dr + 1.0f;
		
		// scale and rotate the point
		float zr = powf( r,power);
		theta = theta*power;
		phi = phi*power;
		
		// convert back to cartesian coordinates
		z = Vec3(zr) * Vec3(sinf(theta)*cosf(phi), sinf(phi)*sinf(theta), cosf(theta));
		z = z + pos;
	}
	return 0.5f*logf(r)*r/dr;
}

// sneak in some double
__global__ void computeUV(int height, int width, float *uvxArr, float *uvyArr)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;

    float uvx = (tanf(3.14159265f / 4.0f)) * (float)(2*x - width) / (float) width;
    float uvy = (tanf(3.14159265f / 4.0f)) * ((float) height / (float) width) * (2.f*(float)y-(float) height) / (float) height;
    uvxArr[(y*width)+x] = uvx;
    uvyArr[(y*width)+x] = uvy;
}

// take a look at what operations are done in "de"
// execution divergence
__global__ void trace(int height, int width, float *uvxArr, float *uvyArr, float *distance, Vec3 lookDirection, Vec3 camUp, Vec3 camRight, Vec3 cameraLocation)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //const int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = 0;//i % width;
    unsigned int y = 0;//i / width; 
    indToZorder(x,y,i);
    if(x >= width)
        return;
    if(y >= height)
        return;

    float uvx = uvxArr[(y*width)+x];
    float uvy = uvyArr[(y*width)+x];
	Vec3 rayDirection = Vec3::normalize(lookDirection + Vec3(uvx) * camUp + Vec3(uvy) * camRight);
    float totalDistance = 0;
    bool hit = false;;
    for(int iter= 0; iter < 128; iter++)
    {
        Vec3 p = cameraLocation + Vec3(totalDistance) * rayDirection;
        
        float currentDist = de(p);
        totalDistance += currentDist;
        if (totalDistance > 10) {
    	    totalDistance = INFINITY;
    	    break;
        }
        if (currentDist < 0.00001f) {
    	    hit = true;
    	    break;
        }
    }
    distance[(y*width)+x] = totalDistance;
}

// flip indexes
// this will compute the shading for all the pixels in the ouput image
__global__ void shade(int height, int width, float *uvxArr, float *uvyArr, float *distance, 
    Vec3 lookDirection, Vec3 camUp, Vec3 camRight, Vec3 backgroundColor, Vec3 cameraLocation, 
    float *rawR, float *rawG, float *rawB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;
        /*
    float uvx = __ldg(&uvxArr[(y*width)+x]);
    float uvy = __ldg(&uvyArr[(y*width)+x]);
    float dist = __ldg(&distance[(y*width)+x]);
    */
    float uvx = uvxArr[(y*width)+x];
    float uvy = uvyArr[(y*width)+x];
    float dist = distance[(y*width)+x];

    Vec3 ret = backgroundColor;
    if(dist != INFINITY)
    {
        dist-=0.0001f;
    	Vec3 rayDirection = Vec3::normalize(lookDirection + Vec3(uvx) * camUp + Vec3(uvy) * camRight);

        Vec3 hitPoint = cameraLocation + Vec3(dist) * rayDirection;
        Vec3 normal = getNormal(hitPoint);

        float lamb = 0.6;
        float spec = 0.2;
        Vec3 objectColor = Vec3(0.8,0.2,0.8);

        Vec3 toLight = ( Vec3(2,2,1) - hitPoint);
		toLight = Vec3::normalize(toLight);

		Vec3 lambIn = Vec3(lamb) * fabsf(Vec3::dot(normal, toLight));
            
		Vec3 specIn = Vec3(spec) * powf(fabsf(Vec3::dot(normal, Vec3::normalize(toLight - hitPoint))), 1);
		ret = clamp(((lambIn  * clamp(objectColor)) + specIn));

    }
    rawR[(y*width)+x] = ret.x;
    rawG[(y*width)+x] = ret.y;
    rawB[(y*width)+x] = ret.z;
    
}


__global__ void globalIllumination(int height, int width, float *uvxArr, float *uvyArr, float *distance, 
    Vec3 lookDirection, Vec3 camUp, Vec3 camRight, Vec3 backgroundColor, Vec3 cameraLocation, 
    float *rawR, float *rawG, float *rawB)
{
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;
    float uvx = uvxArr[(y*width)+x];
    float uvy = uvyArr[(y*width)+x];
    float dist = distance[(y*width)+x];

    if(dist != INFINITY)
    {
        float gi;
        Vec3 rayDirection = Vec3::normalize(lookDirection + Vec3(uvx) * camUp + Vec3(uvy) * camRight);
        Vec3 hitPoint = cameraLocation + Vec3(dist) * rayDirection;
        Vec3 normal = getNormal(hitPoint);
        
        float totalDistance = 0;
        for(int i = 0; i < 10; i++)
        {
            Vec3 p = hitPoint + Vec3(totalDistance) * normal;
            float currentDist = de(p);
            totalDistance += currentDist;
        }
        gi = normalize(0.001,0,totalDistance);
        if (gi > 1)
            gi = 1;
        gi = 1-gi;
        rawR[(y*width)+x] *= gi;
        rawG[(y*width)+x] *= gi;
        rawB[(y*width)+x] *= gi;

    }

}

// shared memory
__global__ void downsample(int height, int width, int scale, float *rawR, float *rawG, float *rawB, float *imageR, float *imageG, float *imageB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;

    int outX = x/scale;
    int outY = y/scale;
    atomicAdd(&imageR[(outY*(width/scale))+outX],rawR[((y)*width)+(x)] / (float)(scale*scale));
    atomicAdd(&imageG[(outY*(width/scale))+outX],rawG[((y)*width)+(x)] / (float)(scale*scale));
    atomicAdd(&imageB[(outY*(width/scale))+outX],rawB[((y)*width)+(x)] / (float)(scale*scale));
}

int main() {
    int width = 2560;
    int height = 1440;

    int scale = 1;

    int heightScale = scale*height;
    int widthScale = scale*width;

    float *rawR;
    float *rawG;
    float *rawB;
    float *uvxArr;
    float *uvyArr;
    float *imageR;
    float *imageG;
    float *imageB;
    float *distance;


    cudaMalloc(&rawR, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&rawG, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&rawB, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&uvxArr, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&uvyArr, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&distance, (width*scale)*(height*scale) * sizeof(float));
    
    cudaMallocManaged(&imageR, width*height * sizeof(float));
    cudaMallocManaged(&imageG, width*height * sizeof(float));
    cudaMallocManaged(&imageB, width*height * sizeof(float));
    
    float max = 0;
    float min = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            imageR[((y)*width)+(x)] = 0;
            imageG[((y)*width)+(x)] = 0;
            imageB[((y)*width)+(x)] = 0;
        }
    }

    Vec3 cameraLocation = {1.2f,1.2f,1.2f};
    Vec3 focus = {0,0,0};
    Vec3 worldUp = Vec3(0,1,0);
    Vec3 lookDirection = Vec3::normalize(focus - cameraLocation);
    Vec3 camUp = Vec3::normalize(Vec3::cross(worldUp, lookDirection));
    Vec3 camRight = Vec3::normalize(Vec3::cross(lookDirection, camUp));

    dim3 threadsScale;
    dim3 blocksScale;

    threadsScale.x = 32;
    threadsScale.y = 32;

    blocksScale.x = 1+ (widthScale/threadsScale.x);
    blocksScale.y = 1+ (heightScale/threadsScale.y);


    // compute uvxy

    getErrorCuda((computeUV<<<blocksScale, threadsScale>>>(heightScale, widthScale, uvxArr, uvyArr)));

    // trace

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

    dim3 threadsTrace;
    dim3 blocksTrace;

    threadsTrace.x = 1024;
    blocksTrace.x = 1+ ((widthScale*heightScale)/threadsTrace.x);

    getErrorCuda((trace<<<blocksTrace, threadsTrace>>>(heightScale, widthScale, uvxArr, uvyArr, distance, lookDirection, camUp, camRight, cameraLocation)));

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Trace time: " << elapsed_seconds.count() << std::endl;


    // shade
    Vec3 backgroundColor = {0.3f};
    getErrorCuda((shade<<<blocksScale, threadsScale>>>(heightScale, widthScale, uvxArr, uvyArr, distance, 
    lookDirection, camUp, camRight, backgroundColor, cameraLocation, rawR, rawG, rawB)));

    // global illumination
    getErrorCuda((globalIllumination<<<blocksScale, threadsScale>>>(heightScale, widthScale, uvxArr, uvyArr, distance, 
    lookDirection, camUp, camRight, backgroundColor, cameraLocation, rawR, rawG, rawB)));


    // downsample
    getErrorCuda((downsample<<<blocksScale, threadsScale>>>(heightScale, widthScale, scale, rawR, rawG, rawB, imageR, imageG, imageB)));

    cudaDeviceSynchronize();


    std::cout << "writing image" << std::endl;
    writeImageRGB("test.png", width,height, imageR,imageG,imageB, "output");

    return 0;
}
