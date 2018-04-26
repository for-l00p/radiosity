#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "errorchecking.cu"

#define PATCH_NUM 1
#define SAMPLES 5

struct PerRayData_pathtrace
{
	optix::float3 origin;
	optix::float3 direction;
	optix::float3 result;
	float importance;
	int depth;
};

struct PatchData {
	optix::float3 a;
	optix::float3 b;
	optix::float3 c;
	optix::float3 norm;
	int id;
};

//generates random kernel
__global__ void rand_kernel(curandState *state, int seed) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

/*
uses random kernel to calculate ray direction 
n : number of rays to generate
num : number of faces
faces[] : array containing all the faces struct
*result : pointer to 2d array of Face -> Array of Directions 
*/
__global__ void generate_ray_dir(curandState *rand1, curandState *rand2, int n, PatchData *faces, int num, optix::float3 *dir, optix::float3 *pts) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int count = 0;

	for (int i = 0; i < num; i++) {
		while (count < n) {
			// generating direction for the face
			//curandState localState = rand1[idx];
			//curandState localState2 = rand2[idx];
			//float sin_theta = sqrt(curand_uniform(&localState /*+ idx*/));
			float sin_theta = sqrt(curand_uniform(rand1 + idx));
			float cos_theta = sqrt(1 - sin_theta * sin_theta);
			//float psi = 2 * 3.14159265359 * curand_uniform((&localState  /*+ idx*/));
			float psi = 2 * 3.14159265359 * curand_uniform((rand1  + idx));
			float a1 = sin_theta * cos(psi);
			float b1 = sin_theta * sin(psi);
			float c1 = cos_theta;

			optix::float3 v1 = a1 * (faces[i].a - faces[i].b);
			optix::float3 v2 = b1 * (faces[i].a - faces[i].c);
			optix::float3 v3 = c1 * faces[i].norm;
			dir[i*n + count] = v1 + v2 + v3;
			
			//float r1 = curand_uniform((&localState2  /*+ idx*/));
			float r2 = curand_uniform((rand2 + idx));
			float r1 = curand_uniform((rand2  + idx));
			//float r2 = curand_uniform((&localState2 /*+ idx*/));

			optix::float3 pt = (1.0 - sqrt(r1))*faces[i].a + (sqrt(r1) * (1.0 - r2))*faces[i].b + (r2 * sqrt(r1)*faces[i].c);
			pts[i*n + count] = pt;
			count++;
		}
	}
	count = 0;
}

int main() {
	curandState *d_state;
	cudaMalloc((void**)&d_state, sizeof(curandState));
	CudaCheckError();
	curandState *d_state1;
	cudaMalloc((void**)&d_state1, sizeof(curandState));
	CudaCheckError();
	rand_kernel <<<1, 1 >>>(d_state, 1234);
	CudaCheckError();
	rand_kernel <<<1, 1 >>>(d_state1, 5678);
	CudaCheckError();

	//host data structures
	PatchData test;
	test.a = optix::make_float3(1.0);
	test.b = optix::make_float3(1.0);
	test.c = optix::make_float3(1.0);
	test.norm = optix::make_float3(0.5);
	//PatchData patches[1] = { test };
	PatchData *patches = (PatchData*)malloc(sizeof(PatchData));
	/*PatchData *patches = NULL;
	cudaMalloc((void**)&patches, PATCH_NUM * sizeof(PatchData));*/
	patches[0] = test;

	optix::float3 *c_dir_arr = (optix::float3*)malloc(SAMPLES *PATCH_NUM * sizeof(optix::float3));
	optix::float3 *c_pt_arr = (optix::float3*)malloc(SAMPLES*PATCH_NUM * sizeof(optix::float3));


	//device data structures
	PatchData *g_patch_arr=NULL;
	optix::float3 *g_dir_arr=NULL, *g_pt_arr=NULL;
	cudaMalloc((void**)&g_patch_arr, PATCH_NUM*sizeof(PatchData));
	CudaCheckError();
	//
	cudaMalloc((void**)&g_dir_arr, SAMPLES * PATCH_NUM * sizeof(optix::float3));
	CudaCheckError();
	cudaMalloc((void**)&g_pt_arr, SAMPLES*PATCH_NUM * sizeof(optix::float3));
	CudaCheckError();

	cudaMemcpy( g_patch_arr, patches, PATCH_NUM * sizeof(PatchData), cudaMemcpyHostToDevice);
	CudaCheckError();

	generate_ray_dir <<<1, 1 >>> (d_state, d_state1, SAMPLES, g_patch_arr, PATCH_NUM, g_dir_arr, g_pt_arr);
	CudaCheckError();
	cudaMemcpy(c_dir_arr, g_dir_arr, SAMPLES*PATCH_NUM * sizeof(optix::float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_pt_arr, g_pt_arr, SAMPLES*PATCH_NUM * sizeof(optix::float3), cudaMemcpyDeviceToHost);
	for (int i = 0; i < PATCH_NUM; i++) {
		for (int j = 0; j < SAMPLES; j++) {
			printf("%f \t", c_dir_arr[i*SAMPLES +j]);
		}
		printf("\n");
	}


	

	return 0;
}



