# Simple profiling wiht nvvp

The point with this excersize is to give you a very quick introduction to the Nvidia visual profiler and how to look at the program using it.

1. Compile the program: 'nvcc -lineinfo main.cu' 

2. Run the visual profiler, nvvp.

3. Create a new profiling session

4. Choose the right apllication and start the profiling

Take a look at how it looks in the profiler, figure out how long it takes to transfer the data back and forth to the device and how long the kernel execution takes.