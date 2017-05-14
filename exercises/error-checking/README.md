# Error checking

Start from the provided skeleton code [error-test.cu](error-test.cu) that
provides some convenience macros for error checking. The macros are defined
in the header file [error_check.h](error_check.h). Add the missing memory allocations
and copies and the kernel launch and check that your code works.

1. What happens if you try to launch kernel with too large block size? When do you
catch the error if you remove the `cudaDeviceSynchronize()` call?
2. What happens if you try to dereference a pointer to device memory in host code?
3. What if you try to access host memory from the kernel?

Remember that you can use also cuda-memcheck!

