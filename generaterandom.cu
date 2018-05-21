#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include <cstdio>
#include <time.h>
#include <ctime>
#include <curand_kernel.h>
#include <cuda.h>

#include "errorchecking.cu"


__global__ void initialise_curand_on_kernels3(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ float generate3(curandState* globalState, int ind)
{
	//copy state to local mem
	curandState localState = globalState[ind];
	//apply uniform distribution with calculated random
	float rndval = curand_uniform(&localState);
	//update state
	globalState[ind] = localState;
	//return value
	return rndval;
}

__global__ void set_random_number_from_kernels3(float* _ptr, curandState* globalState, const unsigned int _points)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//only call gen on the kernels we have inited
	//(one per device container element)
	if (idx < _points)
	{
		float x = generate3(globalState, idx);
		printf("float %f  block %d\n", x, blockIdx.x);
		_ptr[idx] = x;
	}
}

int tee() {
	srand(time(NULL));

	//naively setting the threads per block and block per grid sizes, where 100 is the amount of rngs
	int threadsPerBlock = 512;
	int nBlocks = 300 / threadsPerBlock + 1;
	printf("# of blocks", nBlocks);
	//alocate space for each kernels curandState
	curandState* deviceStates;
	cudaMalloc(&deviceStates, nBlocks * sizeof(curandState));
	CudaCheckError();

	//call curand_init on each kernel with the same random seed
	//and init the rng states
	initialise_curand_on_kernels3 << <nBlocks, threadsPerBlock >> > (deviceStates, unsigned(time(NULL)));
	CudaCheckError();


	//allocate space for the device container of rns
	float* d_random_floats;
	cudaMalloc((void**)&d_random_floats, sizeof(float) * 50000);
	CudaCheckError();

	//calculate per element of the container a rn
	set_random_number_from_kernels3 << <nBlocks, threadsPerBlock >> > (d_random_floats, deviceStates, 50000);
	CudaCheckError();
	return 0;
}