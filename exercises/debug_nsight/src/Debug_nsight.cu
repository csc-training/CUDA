#include <random>
#include <iostream>

__global__ void scaleKernel(float *dataIn, float *dataOut, float scale, int count) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx >= count)
		return;

	const float in = dataIn[idx];
	dataIn[idx] = in * scale;
}


int main(void)
{
	int size = 100000;

	float *hostVal = new float[size];
	float *devA;
	float *devB;

	std::default_random_engine generator(1312);
	std::uniform_real_distribution<float> distribution(0.0,10);

	for (int i = 0; i < size; ++i) {
		hostVal[i] = distribution(generator);

	}

	cudaMalloc(&devA, size*sizeof(float));
	cudaMalloc(&devB, size*sizeof(float));

	cudaMemcpy(devA, hostVal, size*sizeof(float), cudaMemcpyDefault);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, 2, size);

	cudaMemcpy(hostVal, devB, size*sizeof(float), cudaMemcpyDefault);

}
