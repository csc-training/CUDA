
all: mandelbrot

pngwriter.o: pngwriter.c pngwriter.h
	nvcc --x=c -O2 -c -o $@ $<

mandelbrot: mandelbrot.cu pngwriter.o
	nvcc -O3 -arch=sm_35 --relocatable-device-code=true -Xcompiler -Wall -Xcompiler -fopenmp $^ -o $@ -lpng

clean:
	rm -f *.o mandelbrot
