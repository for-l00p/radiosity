#include <optixu/optixu_math_namespace.h>
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

#define PATCH_NUM 5
#define SAMPLES 10

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
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ void intersectAllTriangles(const Ray& r, const int number_of_triangles, PatchData *faces, int &face_id) {

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


/*
uses random kernel to calculate ray direction
n : number of rays to generate
num : number of faces
faces[] : array containing all the faces struct
*result : pointer to 2d array of Face -> Array of Directions
*/
__global__ void generate_ray_dir(curandState *rand1, curandState *rand2, int n, PatchData *faces, int num, int *hit) {

	//printf("num %d", num);

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int i = blockIdx.x;
	int count = 0;
	/*
	for (int i = 0; i < num; i++) {
	while (count < n) {*/
	float sin_theta = sqrt(curand_uniform(rand1 + idx));
	float cos_theta = sqrt(1 - sin_theta * sin_theta);
	float psi = 2 * 3.14159265359 * curand_uniform((rand1 + idx));
	float a1 = sin_theta * cos(psi);
	float b1 = sin_theta * sin(psi);
	float c1 = cos_theta;


	optix::float3 v1 = a1 * (faces[i].b - faces[i].a);
	optix::float3 v2 = b1 * (faces[i].c - faces[i].a);
	optix::float3 v3 = c1 * faces[i].norm;

	curandState localState = rand1[idx];
	float r2 = curand_uniform(&localState);
	//printf("index %d \n", idx);
	printf("random float %f \n", r2);
	float r1 = curand_uniform((rand2 + int(idx/2)));
	//printf("random float2 %f \n", r1);

	optix::float3 pt = (1.0 - sqrt(r1))*faces[i].a + (sqrt(r1) * (1.0 - r2))*faces[i].b + (r2 * sqrt(r1)*faces[i].c);
	optix::float3 direction = v1 + v2 + v3;
	//printf("dir (%f, %f, %f) \n", direction.x, direction.y, direction.z);
	Ray ray = Ray(pt, direction);
	int face = 0;
	intersectAllTriangles(ray, num, faces, face);
	//printf("face %d \n", face);
	hit[idx] = face;
	rand1[idx] = localState;

}

int main() {
	curandState *d_state;
	cudaMalloc((void**)&d_state, sizeof(curandState));
	CudaCheckError();
	curandState *d_state1;
	cudaMalloc((void**)&d_state1, sizeof(curandState));
	CudaCheckError();
	srand(time(NULL));
	rand_kernel << <1, 1 >> >(d_state, rand());
	CudaCheckError();
	srand(time(NULL));
	rand_kernel << <1, 1 >> >(d_state1, rand());
	CudaCheckError();

	//host data structures
	PatchData test;
	PatchData *patches = (PatchData*)malloc(PATCH_NUM * sizeof(PatchData));
	int p = 0;
	for (p = 0; p < PATCH_NUM; p++) {
		PatchData *t = (PatchData*)malloc(sizeof(PatchData));
		t->a = optix::make_float3(1.0f, 1.0f, 1.0f);
		t->b = optix::make_float3(1.0f, 0.0f, 1.0f);
		t->c = optix::make_float3(1.0f, 1.0f, 0.0f);
		t->norm = optix::make_float3(0.0f, -1.0f, 0.0f);
		patches[p] = *t;
	}

	//device data structures
	PatchData *g_patch_arr = (PatchData*)malloc(PATCH_NUM * sizeof(PatchData));
	optix::float3 *g_dir_arr = (optix::float3*)malloc(SAMPLES*PATCH_NUM * sizeof(optix::float3)), *g_pt_arr = (optix::float3*)malloc(SAMPLES*PATCH_NUM * sizeof(optix::float3));
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

	start = std::clock();

	generate_ray_dir << <PATCH_NUM, 1024 >> > (d_state, d_state1, SAMPLES, g_patch_arr, PATCH_NUM, g_hit);
	CudaCheckError();
	cudaMemcpy(c_hit, g_hit, SAMPLES*PATCH_NUM * sizeof(int), cudaMemcpyDeviceToHost);
	duration = (std::clock() - start) / (float)CLOCKS_PER_SEC;
	printf("%f\n", duration);
	for (int i = 0; i < PATCH_NUM*SAMPLES; i++) {
		printf("face %d \n", c_hit[i]);
	}
	printf("Done");

	free(patches);

	return 0;
}


