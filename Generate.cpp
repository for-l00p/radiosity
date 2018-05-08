#include <optixu/optixu_math_namespace.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <ctime>

#define PATCH_NUM 1000
#define SAMPLES 1024


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

/*
uses random kernel to calculate ray direction
n : number of rays to generate
num : number of faces
faces[] : array containing all the faces struct
*result : pointer to 2d array of Face -> Array of Directions
*/
void generate_ray_dir( int n, PatchData *faces, int num, optix::float3 *dir, optix::float3 *pts) {


	int count = 0;

	for (int i = 0; i < num; i++) {
		int poop = 0;
		while (count < n) {
			float sin_theta = sqrt((float)rand() / ((float)RAND_MAX + 1));
			float cos_theta = sqrt(1 - sin_theta * sin_theta);

			float psi = ((float)rand() / ((float)RAND_MAX + 1) * 2 * 3.14159265359);

			float a1 = sin_theta * cos(psi);
			float b1 = sin_theta * sin(psi);
			float c1 = cos_theta;


			optix::float3 v1 = a1 * (faces[i].b - faces[i].a);
			optix::float3 v2 = b1 * (faces[i].c - faces[i].a);
			optix::float3 v3 = c1 * faces[i].norm;

			float r1 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 1.0));
			float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 1.0));
		/*	float r2 = curand_uniform((rand2 + idx));
			float r1 = curand_uniform((rand2 + idx));*/

			optix::float3 pt = (1.0 - sqrt(r1))*faces[i].a + (sqrt(r1) * (1.0 - r2))*faces[i].b + (r2 * sqrt(r1)*faces[i].c);
			optix::float3 direction = v1 + v2 + v3;
			dir[i*n + count] = direction; //optix::make_float3(0.5f, 0.5f, 2.5* count); //
			int pos = i * n + count;
			optix::float3 pt2 = pt;//optix::make_float3(0.5f, 0.5f, count);
			pts[pos] = pt2;//
			count++;
		}
		count = 0;
	}
}

int pop() {
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

	std::clock_t start;
	float duration;

	start = std::clock();

	generate_ray_dir(SAMPLES, patches, PATCH_NUM, g_dir_arr, g_pt_arr);
	duration = (std::clock() - start) / (float)CLOCKS_PER_SEC;

	std::cout << "printf: " << duration << '\n';
	float f = g_pt_arr[(PATCH_NUM - 1)*SAMPLES + SAMPLES - 1].z;
	float f2 = g_dir_arr[(PATCH_NUM - 1)*SAMPLES + SAMPLES - 1].z;
	//for (int i = 0; i < PATCH_NUM; i++) {
	//	for (int j = 0; j < SAMPLES; j++) {
	//		float f = g_pt_arr[i*SAMPLES + j].z;
	//		float f2 = g_dir_arr[i*SAMPLES + j].z;
	//		printf("%f , %f \n", f, f2);
	//	}
	//	printf("\n");
	//}
	printf("Done");

	free(patches);
	return 0;
}



