# Volume ray tracing

The program is a specialized raytracer for fractal geometry. It builds on the principle of ray marching [1], since we can't compute the interaction of a ray with the fractal geometry there is no way to get the exact intersection point. We can however compute the distance from a given point to the closest place on the fractal geometry using the distance estimator function (DE in main.cu). Based on the distance to the geometry we know we can move all the rays that far into the scene safely, we then continue to check the distance estimation function at the new place and move forward, we do this until we are sufficiently close to the geometry.

The raytracer also computes shading for the rays that hit, a quick and dirty global illumination lighting, and finally it scales down the image to give a better image quality.

The main goal is that you are able to identify the issues in the code, the code is unnecessary split into smaller kernels all doing just a part of the computation to make it easier to locate the issues in the code. If you have time you can try to fix some of the issues in the code, however that is not required, note that some of them are a lot more complex that what you probably have time for now.

Use nvvp to identify the main issues in the code that affect the performance. Even though the majority of the time is spent on the trace kernel and realistically that is the one that needs the most optimization, look at the other kernels, most of them also have some issues.

Build the program on taito-gpu: ```nvcc -O3 --std=c++11 -arch=sm_35 -lineinfo main.cu -lpng```

Recommended practice is profile it on taito and then move the results back to your own machine and look on them there.

To make things easier there are the metrics and timeline files included here.

However if you want to test it out yourself, or test any improvements you made, you can generate the files used by:

```srun --gres=gpu:1 -pgpu  --time=00:15:00 --mem=12000 nvprof --analysis-metrics -o metrics.nvvp ./a.out ```

```srun --gres=gpu:1 -pgpu  --time=00:15:00 --mem=12000 nvprof -o timeline.nvvp ./a.out ```

Then transfer the output files back to your own machine use SCP.

Assuming you cloned the exercises into your home directory the command would be :

```scp <username>@taito-gpu.csc.fi:CUDA/exercises/kernel-optimization/timeline.nvvp . ```

```scp <username>@taito-gpu.csc.fi:CUDA/exercises/kernel-optimization/metrics.nvvp . ```

Run the command on the LOCAL machine and replace ```<username>``` with your username on taito.

To import the data into nvvp on your local machine: file -> import -> nvprof -> single process, for the timeline file put timeline.nvvp and for metrics put metrics.nvvp both should be located in the exercise folder.


[1] http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
