#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include <cstdio>
#include <time.h>
#include <ctime>
#include <optixu/optixu_math_namespace.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "errorchecking.cu"
//
//#define PATCH_NUM 512
//#define SAMPLES 512

struct PatchData {
	optix::float3 a;
	optix::float3 b;
	optix::float3 c;
	optix::float3 norm;
	int id;

};

extern int* main_test(PatchData *patches, int PATCH_NUM, int SAMPLES);
struct Ray {
	optix::float3 orig;	// ray origin
	optix::float3 dir;		// ray direction	
	__device__ Ray(optix::float3 o_, optix::float3 d_) : orig(o_), dir(d_) {}
};

__device__ float RayTriangleIntersection(const Ray &r,
	const optix::float3 &v0, const optix::float3 &edge1, const optix::float3 &edge2) {

	optix::float3 tvec = r.orig - v0;
	optix::float3 pvec = optix::cross(r.dir, edge2);
	float  det = optix::dot(edge1, pvec);

	det = __fdividef(1.0f, det);  // CUDA intrinsic function 

	float u = optix::dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	optix::float3 qvec = optix::cross(tvec, edge1);

	float v = optix::dot(r.dir, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f)
		return -1.0f;

	return optix::dot(edge2, qvec) * det;
}

//generates random kernel
__global__ void rand_kernel(curandState *state, int seed) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed, idx, 0, &state[0]);
}

__device__ void intersectAllTriangles(Ray& r, const int number_of_triangles, PatchData *faces, int &face_id) {

	float max_dist = 10000.0f;
	int max_face = -1;
	for (int i = 0; i < number_of_triangles; i++)
	{
		optix::float3 v0 = faces[i].a; 
		optix::float3 edge1 = (faces[i].b - faces[i].a); 
		optix::float3 edge2 = (faces[i].c - faces[i].a); ;

														 // intersect ray with reconstructed triangle	
		float dist = RayTriangleIntersection(r,v0, edge1, edge2);

		// keep track of closest distance and closest triangle
		// if ray/tri intersection finds an intersection point that is closer than closest intersection found so far
		if (dist < max_dist && dist > 0.001)
		{
			max_dist = dist;
			max_face = i;
		}
	}
	face_id = max_face;
	//printf("face_id %d \n", face_id);
}

__global__ void initialise_curand_on_kernels(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ float generate(curandState* globalState, int ind)
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

__global__ void set_random_number_from_kernels(float* _ptr, curandState* globalState, const unsigned int _points)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//only call gen on the kernels we have inited
	//(one per device container element)
	if (idx < _points)
	{
		float x = generate(globalState, idx);
		//printf("%f \n", x);
		_ptr[idx] = x;
	}
}

/*
uses random kernel to calculate ray direction
n : number of rays to generate
num : number of faces
faces[] : array containing all the faces struct
*result : pointer to 2d array of Face -> Array of Directions
*/
__global__ void generate_ray_dir(curandState*globalState, PatchData *faces, int num, int *hit) {

	for (int h = 0; h < 32; h++) {
		int idx = threadIdx.x + blockDim.x*blockIdx.x + h;
		//int sad = //threadIdx.x + h;
		int i = blockIdx.x;

		float sin_theta = sqrt(generate(globalState, idx));
		float cos_theta = sqrt(1 - sin_theta * sin_theta);
		float psi = 2 * 3.14159265359 * generate(globalState, idx);
		float a1 = sin_theta * cos(psi);
		float b1 = sin_theta * sin(psi);
		float c1 = cos_theta;


		optix::float3 v1 = a1 * (faces[i].b - faces[i].a);
		optix::float3 v2 = b1 * (faces[i].c - faces[i].a);
		optix::float3 v3 = c1 * faces[i].norm;

		float r2 = generate(globalState,idx);
		float r1 = generate(globalState, idx);

		optix::float3 pt = (1.0 - sqrt(r1))*faces[i].a + (sqrt(r1) * (1.0 - r2))*faces[i].b + (r2 * sqrt(r1)*faces[i].c);
		optix::float3 direction = v1 + v2 + v3;

		Ray ray = Ray(pt, direction);
		int face = 0;
		intersectAllTriangles(ray, num, faces, face);
		//printf("face %d for core %d r2 %f\n", face, blockIdx.x, r2);
		if (threadIdx.x == 0 && blockIdx.x == 0) {
		
			printf("%d %d %d %d\n", blockIdx.x, threadIdx.x, h, face);
		}
		if (face == -1 && threadIdx.x == 0 && blockIdx.x==0) {
			printf("missed %d %d %d\n", blockIdx.x, threadIdx.x, h);
			h -= 1;
		}
		if (blockIdx.x == 100) {
			printf("random # %f", r2);
		}
		else {
			hit[idx] = face;
		}

	}

}

int* main_test(PatchData *patches, int PATCH_NUM, int SAMPLES) {
	PatchData *g_patch_arr = (PatchData*)malloc(PATCH_NUM * sizeof(PatchData));
	//optix::float3 *g_dir_arr = (optix::float3*)malloc(SAMPLES*PATCH_NUM * sizeof(optix::float3)), *g_pt_arr = (optix::float3*)malloc(SAMPLES*PATCH_NUM * sizeof(optix::float3));
	cudaMalloc((void**)&g_patch_arr, PATCH_NUM * sizeof(PatchData));
	CudaCheckError();

	int* g_hit = (int*)malloc(SAMPLES*PATCH_NUM * sizeof(int));
	cudaMalloc((void**)&g_hit, SAMPLES*PATCH_NUM * sizeof(int));
	CudaCheckError();

	int *c_hit = (int*)malloc(SAMPLES*PATCH_NUM * sizeof(int));
	cudaMemcpy(g_patch_arr, patches, PATCH_NUM * sizeof(PatchData), cudaMemcpyHostToDevice);
	CudaCheckError();
	std::clock_t start;
	float duration;


	curandState* deviceStates;
	printf("size of curand %d\n", sizeof(curandState));
	cudaMalloc((void**)&deviceStates, PATCH_NUM * sizeof(curandState));
	CudaCheckError();

	initialise_curand_on_kernels << <PATCH_NUM, SAMPLES / 16 >> > (deviceStates, unsigned(time(NULL)));
	cudaDeviceSynchronize();

	CudaCheckError();

	start = std::clock();

	generate_ray_dir << <PATCH_NUM, SAMPLES/16 >> > (deviceStates, g_patch_arr, PATCH_NUM, g_hit);
	cudaDeviceSynchronize();
	CudaCheckError();

	cudaMemcpy(c_hit, g_hit, SAMPLES*PATCH_NUM * sizeof(int), cudaMemcpyDeviceToHost);
	CudaCheckError();

	duration = (std::clock() - start) / (float)CLOCKS_PER_SEC;
	printf("%f\n", duration);

	cudaFree(g_patch_arr);
	CudaCheckError();
	/*cudaFree(g_dir_arr);
	CudaCheckError();*/
	cudaFree(deviceStates);
	CudaCheckError();
	cudaFree(g_hit);
	CudaCheckError();
	return c_hit;
}


