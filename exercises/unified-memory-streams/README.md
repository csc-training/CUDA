# Unified memory with streams


In this exercise we will execute a simple kernel on the GPU that adds up two
arrays. Here we will use multiple streams to do the addition, and will
practice how to separate up managed memory allocation to different
streams. Look at the code in [streams.cu](streams.cu) and complete the TODO's
