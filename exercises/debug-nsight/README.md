# Debugging with nsight

The goal here is to give you a quick introduction to the nsight IDE and how it can be used to debug your cuda programs.

1. Start nsight, works a lot easier on the local machines.

2. Import the project provided using nsight.
    - File -> Import -> General -> Existing projects into workspace
    - Choose the current directory as root and import
    - ASK IF YOU HAVE ISSUES IMPORTING IT !!!

3. Run the program in debugging mode, you may need to change to the debugging mode mnaually.

Figure out what value thread 824 reads from memory, thread 56 of block 3. Or if you cannot find it look at any other thread.

Hint: use breakpoints to stop the program execution.