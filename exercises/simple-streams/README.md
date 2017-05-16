# Streams

In this exercise we will implement a simple copy-in, compute, copy-out test and
examine how the streams can be used to overlap copying and computing. Start
from the provided skeleton and implement following parts:

1. Implement the memory allocations so that host memory allocations are page-locked.
2. Add timing events as in previous exercises.
3. Add the missing parts to the `streamtest` function.

Check the execution profile using `nvvp`.
