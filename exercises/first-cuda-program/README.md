#First cuda program

In this exercise we will write a simple cuda program that adds two
arrays with each other. Take a look at the file "add.cu", that
includes code for adding the two arrays on the CPU. The task is to
implement the same operation on the GPU.  Here we will complete the
non-function code by doing these steps (a TODO exists for each step):

1. Allocate memory for two arrays in device memory
2. Complete the kernel code. The kernel assigns the global thread index to each element in the vector
3. Call the kernel with two arguments, pointer to the allocated device memory and the
length of the array
4. Copy the result vector from device memory to host memory buffer
5. Print out the values for checking

Pay close attention to the kernel call parameters, block and grid sizes!