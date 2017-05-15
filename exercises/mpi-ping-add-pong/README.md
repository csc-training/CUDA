# MPI ping pong 

The first task is to make sure that each rank selects its own GPU device in
[pingpong.cpp](pingpong.cpp).

Rank 0 sends an array filled with `1` to rank 1, which adds `1` the the values
and sends the data back. The code already implements a CPU version. Complete
also the GPU version using CUDA aware MPI.

Bonus assignment: Implement a manual version that is not using cuda aware MPI.

