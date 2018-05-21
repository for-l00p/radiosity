#pragma once
#include <optixu/optixu_math_namespace.h>

#ifndef PatchData
#define PatchData

struct PatchData {
	optix::float3 a;
	optix::float3 b;
	optix::float3 c;
	optix::float3 norm;
	int id;

};


#endif