# Multi GPU vector add

The idea is that you should modify the code so that the add kernel will be split and run on two or more devices.

You need to enumerate the devices on the current node first, allocate memory on them, move parts of the memory to each, start the kernels on all of them, and finally move the data back.

Make sure that each kernel only accesses its own memory space.

You can verify that the kernels run on both GPUs the same time using the visual profiler.

To get more gpus for your runs on taito you need to increase the number of GPUs you allocate for your job by increasing the number in the gres parameter: `--gres=gpu:1` for one GPU,  `--gres=gpu:2` for two GPUs and so on.