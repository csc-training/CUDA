NVCC = nvcc
MPICXX = mpicxx
MPICXXFLAGS = -g -O2 -Wall
LD = $(MPICXX)
NVCFLAGS = -c -O2 -arch=sm_35
LDFLAGS = -L$(CUDADIR)/lib64
LIBS = -lcudart
INCLUDES = -I. -I$(CUDADIR)/include
CUDASRC = add_kernel.cu
MPISRC = pingpong.cpp 
OBJS = $(MPISRC:.cpp=.cpp.o) $(CUDASRC:.cu=.cu.o)
EXE = pingpong


.SUFFIXES:


all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDFLAGS) -o $(EXE) $(OBJS) $(LIBS)
	cp $(EXE) $(WRKDIR)

%.cu.o:%.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -o $@ $<

%.cpp.o:%.cpp
	$(MPICXX) $(MPICXXFLAGS) $(INCLUDES) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) *~ $(EXE) *.o

