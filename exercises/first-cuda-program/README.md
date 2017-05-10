# First cuda program

In this exercise we will write a simple cuda program that sets the
value of an array: `A[i] = i`. Take a look at the file [set.cu](set.cu), that
includes a skeleton of the code. Here we will complete the code by
completing these steps (a TODO exists for each step):

1. Allocate memory for the device array `d_A`
2. Free memory for the device array `d_A`
3. Complete the kernel code. The kernel assigns the global thread index to each element in the vector
4. Call the kernel with two arguments, pointer to the allocated device memory and the
length of the array.
5. Copy the result vector from device memory to host memory buffer

Pay close attention to the kernel call parameters, block and grid
sizes! Can you write the kernel so that it functions even if you
launch too many threads?
