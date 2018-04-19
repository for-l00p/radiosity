#include <optixu/optixu_math_namespace.h>
#include <stdlib.h>


using namespace optix;

struct PerRayData_pathtrace
{
	float3 origin;
	float3 direction;
	float3 result;
	float importance;
	int depth;
};

float3 getCosineDistributionVector(float3 a, float3 b, float3 c, float3 norm) {
	float sin_theta = sqrt((double)rand() / ((double)RAND_MAX + 1));
	float cos_theta = sqrt(1 - sin_theta * sin_theta);

	float psi = ((double)rand() / ((double)RAND_MAX + 1) * 2 * 3.14159265359);

	float3 a1 = float3(sin_theta) * cos(psi);
	float3 b1 = float3(sin_theta) * sin(psi);
	float3 c1 = float3(cos_theta);


	float3 v1 = a1 * (a - b);
	float3 v2 = b1 * (a - c);
	float3 v3 = c1 * norm;

	return v1 + v2 + v3;
}



