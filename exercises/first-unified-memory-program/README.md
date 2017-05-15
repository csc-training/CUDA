# First cuda program using unified memory

In this exercise we will take a simple program [set.cu](set.cu) that uses
device and host memory, and port it to use unified memory. The program sets
the value of an array on the device, `A[i] = i`, and transfers the data to the
host that prints out its values. 

Pay close attention to how and when you access the unified memory on the host
and device. 
