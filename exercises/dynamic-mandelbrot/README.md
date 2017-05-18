# Mandelbrot set 

In this exercise we take a look at Mandelbrot's set. The example code
implements the recursive Mariani-Silver algorithm (see this blog post
for detailed description of the algorithm: [https://devblogs.nvidia.com/parallelforall/introduction-cuda-dynamic-parallelism/](https://devblogs.nvidia.com/parallelforall/introduction-cuda-dynamic-parallelism/))

The provided code example can be compiled using `make` command. Run and profile the
code. Note that you can skip the output writing if you compile the code with `make WRITE_PNG=no`.

If the X11 connection seems too slow, you can save the timeline profiling results in Taito using command
```
srun -pgpu --gres=gpu:1 --mem=5G -n1 --reservation=cuda_fri nvprof -f -o timeline.nvvp ./mandelbrot
```
and then copy the result file `timeline.nvvp` back to your local workstation. See the optimization exercises
for instruction on importing the results to `nvvp`.

Try different values for maximum recursion depth (`MAX_DEPTH`) and initial
number of subdivisions (`INIT_SUBDIV`). Based on the profiling results, can you
explain why deeper recursion depth and/or larger number of initial subdivision
give faster results?

Hint: take a look at the cumulative time spent on different kernels.
