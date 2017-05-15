The program is a specialized raytracer for fractal geometry. It builds on the principle of ray marching [1], since we can't compute the interection of a ray with the fractal geometry there is no way to get the exact intersection point. We can however compute the distance from a given point to the closest place on the fractal geometry using the distance estimator function (DE in main.cu). 



The main goal is that you are able to identify the issues in the code, the code is unsesseseraly split into smaller kernels all doing just a part of the computation to make it easier to locate the issues in the code. If you have time you can try to fix some of the issues in the code, note that some of them are a lot more complex that what you probably have time for now.

Build the program 
    on taito-gpu: nvcc -O3 --std=c++11 -arch=sm_35 main.cu -lpng
    on the local machine:

Use nvvp to idedntify the main issues in the code that affect the performance, all kernels have at least one, some issues are shared between kernels.

[1] http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/