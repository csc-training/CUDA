# Introduction to CUDA programming

17-19th May 2017, at CSC - IT Center for Science

## Table of content
 1. [Schedule](#schedule)
 2. [How to run the exercises](#how-to-run-exercises)
 3. [Exercises](/exercises)
 4. [Slides](intro-to-cuda-csc.pdf)

## Schedule

| Day 1 |                      |
|-------|----------------------|
|9:00 - 9:30   | Introduction to GPUs |
|9:00 - 9:30   | Introduction to GPU programming | 
|9:30 - 10:30  | CUDA Programming 1 |            
|10:30 - 10:45 | Coffee Break        |                        
|10:45 - 11:15 | Exercises |                       
|11:15 - 12:00 | Tools |                               
|12:00 - 12:45 | Lunch  |                                     
|12:45 - 13:15 | Exercises |
|13:15 - 14:00 | Cuda Programming II|
|14:00 - 14:30 | Exercises |
|14:30 - 14:45 | Coffee |
|14:00 - 16:00 | Exercises |
                                                          
                                                          
|Day 2 |           |
|------|-----------|                                                    
|9:00  - 10:00 | CUDA Programming III |
|10:00 - 10:15 | Coffee                |        
|10:15 - 12:00 | Exercises              |                     
|12:00 - 12:45 | Lunch                   |                    
|12:45 - 13:45 | Kernel optimization |                   
|13:45 - 14:30 | Exercises            |                       
|14:30 - 14:45 | Coffee Break          |                      
|14:45 - 15:15 | CUDA Programming IV |                   
|15:15 - 16:00 | Exercises            |                       
                                                          
                                                          
|Day 3         |   |
|----|----|                                            
|9:00  - 9:45  | Dynamic parallelism |               
|9:45 - 10:15  | Coffee |
|10:15 - 11:30 | Exercises |       
|11:30 - 12:00 | Multi-GPU programming |
|12:00 - 13:00 | Lunch                 |                     
|13:00 - 14:30 | Exercises              |                     
|14:30 - 15:00 | MPI and CUDA |                    
|15:00 - 16:00 | Exercises     |                        

## How to run exercises

All exercises except the multi-GPU and MPI labs can be done using the local
classroom workstations. You may also use the
[Taito-GPU](https://research.csc.fi/taito-gpu) partition of Taito cluster.

### Downloading the exercises

The exercises are in this repository in two different folders. In
[exercises](/exercises/) are a set of exercises prepared by CSC.

To get a local copy of the exercises you can clone the repository
```
git clone https://github.com/csc-training/cuda.git
```

This command will create a folder called ```CUDA``` where all the
materials are located. If the repository is updated during the course
you can update your local copy of it with a ```git pull``` command.

### Running on local desktop computers

Classroom workstations have Quadro K600 GPUs and CUDA SDK 8.0. The GPUs have a compute capability of 3.0

#### Compiling

Use ```nvcc -arch=sm_30 <source>.cu``` to compile your program.

Useful flags:

| Flag               | Description                                                                                                                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| ´-arch=sm_xx´        | Shorthand for comping for specific architecure, replace xx with the compute capability targeted                             |
| ´-lineinfo´          | Add the source code into the ptx output, enables mapping from ptx to kernel code, needed for better debugging and profiling |
| ´-Xptxas="-v"´       | Make the ptx assembler output the register and shared memory usage for the compiled file                                    |
| ´-Xptxas="-dlcm=ca"´ | Will enable loads through L1 cache for the entire compiled file                                                             |
| ´-use_fast_math´     | Replaces some single precision math functions with faster lower precision ones                                              |
| ´-rdc=true´          | Compiles relocatable device code, needed for among other things dynamic parallelism                                         |

### Running on Taito-GPU

You can log into [Taito-GPU](https://research.csc.fi/taito-gpu)
front-end node using ssh command ```ssh -Y trngXXX@taito-gpu.csc.fi```
where ```XXX``` is the number of your training account. You can also
use your own CSC account if you have one.
 
Login node does not have a GPU, so all CUDA programs have to be run
on the compute nodes. Serial jobs can be run with `srun` command like this
```
srun -n1 -pgpu --gres=gpu:1 ./my_program
```
You can extend the default time limit for you job with ```-t``` option.
For example, option ```-t15``` will allow you job to run for 15 minutes.

### Reservations

This couse has a resource reservation that can be used for the exercises.
You can run your job within the reservation with ```--reservation``` option,
such as ```srun --reservation=cuda_wed -n1 -pgpu --gres=gpu:1 ./my_program```.
Names of the reservations for Wednesday, Thursday and Friday are ```cuda_wed```, 
```cuda_thu``` and ```cuda_fri``` respectively.