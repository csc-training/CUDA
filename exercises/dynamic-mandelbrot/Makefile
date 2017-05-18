
WRITE_PNG=yes

ifeq ($(WRITE_PNG),no)
	DFLAGS=-DSKIP_PNG_WRITING
endif

all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	nvcc --x=c -arch=sm_35 -O2 -c -o $@ $<

mandelbrot: mandelbrot.cu pngwriter.o
	nvcc -O3 -arch=sm_35 $(DFLAGS) --relocatable-device-code=true \
		-Xcompiler -Wall -Xcompiler -fopenmp $^ -o $@ -lpng

clean:
	rm -f *.o mandelbrot mandelbrot.png
