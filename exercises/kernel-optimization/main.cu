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
	for (int i = 0; i < 10 ; i++) 
    {
		r = Vec3::length(z);
		if (r>cutoff) break;
		
		float theta = acosf(z.z/r);
		float phi = atanf(z.y);
		dr =  powf( r, power-1.0f)*power*dr + 1.0f;
		
		float zr = powf( r,power);
		theta = theta*power;
		phi = phi*power;
		
		z = Vec3(zr) * Vec3(sinf(theta)*cosf(phi), sinf(phi)*sinf(theta), cosf(theta));
		z = z + pos;
	}
	return 0.5f*logf(r)*r/dr;
}

__global__ void zero(float *arr, int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    arr[i] = 0;
}
__global__ void computeUV(int height, int width, float *uvxArr, float *uvyArr)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;

    float widthFloat = width;
    float heightFloat = height;

    float uvx = (tan(3.14159265 / 4.0)) * (2.0*x - widthFloat) /  widthFloat;
    float uvy = (tan(3.14159265 / 4.0)) * ( heightFloat /  widthFloat) * (2.0*y- heightFloat) /  heightFloat;
    uvxArr[(y*width)+x] = uvx;
    uvyArr[(y*width)+x] = uvy;
}

// take a look at what operations are done in "de"
__global__ void trace(int height, int width, float *uvxArr, float *uvyArr, float *distance, Vec3 lookDirection, Vec3 camUp, Vec3 camRight, Vec3 cameraLocation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
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

__global__ void shade(int height, int width, float *uvxArr, float *uvyArr, float *distance,
                      Vec3 lookDirection, Vec3 camUp, Vec3 camRight, Vec3 backgroundColor, Vec3 cameraLocation,
                      float *rawR, float *rawG, float *rawB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= height)
        return;
    if(y >= width)
        return;

    float uvx = uvxArr[(x*width)+y];
    float uvy = uvyArr[(x*width)+y];
    float dist = distance[(x*width)+y];

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
    rawR[(x*width)+y] = ret.x;
    rawG[(x*width)+y] = ret.y;
    rawB[(x*width)+y] = ret.z;

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
__global__ void downsample(int height, int width, int scale, float *rawR, float *rawG, float *rawB, float *imageR, float *imageG, float *imageB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;

    int inX = x*scale;
    int inY = y*scale;

    float outR = 0;
    float outG = 0;
    float outB = 0;
    for(int i = 0; i < scale; i++)
    {
        for(int j = 0; j < scale; j++)
        {
            outR += (rawR[((inY+i)*width*scale)+(inX+j)]);
            outG += (rawG[((inY+i)*width*scale)+(inX+j)]);
            outB += (rawB[((inY+i)*width*scale)+(inX+j)]);
        }
    }
    imageR[(y*width)+x] = outR/(float)(scale*scale);
    imageG[(y*width)+x] = outG/(float)(scale*scale);
    imageB[(y*width)+x] = outB/(float)(scale*scale);
}


int main() {
    int width = 1024;
    int height = 768;

    int scale = 2;

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
    
    float *d_imageR;
    float *d_imageG;
    float *d_imageB;
    
    
    float *distance;


    cudaMalloc(&rawR, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&rawG, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&rawB, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&uvxArr, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&uvyArr, (width*scale)*(height*scale) * sizeof(float));
    cudaMalloc(&distance, (width*scale)*(height*scale) * sizeof(float));
    
    cudaMalloc(&d_imageR, width*height * sizeof(float));
    cudaMalloc(&d_imageG, width*height * sizeof(float));
    cudaMalloc(&d_imageB, width*height * sizeof(float));

    cudaMemset(rawB, 0, (width*scale)*(height*scale));
    cudaMemset(rawG, 0, (width*scale)*(height*scale));
    cudaMemset(rawB, 0, (width*scale)*(height*scale));
    cudaMemset(uvxArr, 0, (width*scale)*(height*scale));
    cudaMemset(uvyArr, 0, (width*scale)*(height*scale));
    cudaMemset(distance, 0, (width*scale)*(height*scale));
    
    cudaMemset(d_imageR, 0, (width)*(height));
    cudaMemset(d_imageG, 0, (width)*(height));
    cudaMemset(d_imageB, 0, (width)*(height));


    cudaMallocHost(&imageR, width*height * sizeof(float));
    cudaMallocHost(&imageG, width*height * sizeof(float));
    cudaMallocHost(&imageB, width*height * sizeof(float));
    

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


    // compute UV 

    dim3 threadsUv;
    dim3 blocksUv;

    threadsUv.x = 32;
    threadsUv.y = 32;

    blocksUv.x = 1+ (widthScale/threadsUv.x);
    blocksUv.y = 1+ (heightScale/threadsUv.y);

    getErrorCuda((computeUV<<<blocksUv, threadsUv>>>(heightScale, widthScale, uvxArr, uvyArr)));

    // trace

    dim3 threadsTrace;
    dim3 blocksTrace;

    threadsTrace.x = 1024;
    threadsTrace.y = 1;

    blocksTrace.x = 1 + (widthScale/threadsTrace.x);
    blocksTrace.y = 1 + (heightScale/threadsTrace.y);

    getErrorCuda((trace<<<blocksTrace, threadsTrace>>>(heightScale, widthScale, uvxArr, uvyArr, distance, lookDirection, camUp, camRight, cameraLocation)));

    Vec3 backgroundColor = {0.3f};


    // shade
    dim3 threadsShade;
    dim3 blocksShade;

    threadsShade.x = 32;
    threadsShade.y = 32;

    blocksShade.x = 1 + (heightScale/threadsShade.x);
    blocksShade.y = 1 + (widthScale/threadsShade.y);

    getErrorCuda((shade<<<blocksShade, threadsShade>>>(heightScale, widthScale, uvxArr, uvyArr, distance,
    lookDirection, camUp, camRight, backgroundColor, cameraLocation, rawR, rawG, rawB)));

    
    // global illumination

    dim3 threadsGi;
    dim3 blocksGi;

    threadsGi.x = 32;
    threadsGi.y = 32;

    blocksGi.x = 1+ (widthScale/threadsGi.x);
    blocksGi.y = 1+ (heightScale/threadsGi.y);

    getErrorCuda((globalIllumination<<<blocksGi, threadsGi>>>(heightScale, widthScale, uvxArr, uvyArr, distance, 
    lookDirection, camUp, camRight, backgroundColor, cameraLocation, rawR, rawG, rawB)));


    // downsample

    dim3 threadsDs;
    dim3 blocksDs;

    threadsDs.x = 16;
    threadsDs.y = 16;

    blocksDs.x = 1+ (width/threadsDs.x);
    blocksDs.y = 1+ (height/threadsDs.y);
    getErrorCuda((downsample<<<blocksDs, threadsDs>>>(height, width, scale, rawR, rawG, rawB, d_imageR, d_imageG, d_imageB)));

    cudaDeviceSynchronize();

    cudaMemcpy(imageR, d_imageR, sizeof(float)*height*width, cudaMemcpyDefault);
    cudaMemcpy(imageG, d_imageG, sizeof(float)*height*width, cudaMemcpyDefault);
    cudaMemcpy(imageB, d_imageB, sizeof(float)*height*width, cudaMemcpyDefault);

    std::cout << "writing image" << std::endl;
    writeImageRGB("test.png", width,height, imageR,imageG,imageB, "output");

    return 0;
}