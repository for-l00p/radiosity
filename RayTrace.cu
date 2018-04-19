#include <optixu/optixu_math_namespace.h>
//#include "Radiosity.h"

using namespace optix;

struct PerRayData_pathtrace
{
	float3 origin;
	float3 direction;
	int depth;
};


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(int , hit_face, rtIntersection, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, );
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );


RT_PROGRAM void setRequiredValues() {

}

RT_PROGRAM void rayTrace() {

	PerRayData_pathtrace prd;
	prd.seed = make_float3(0.f);
	prd.direction = make_float3(0.f);
	prd.depth = 0;

	for (;;)
	{
		Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, 0.0001f, RT_DEFAULT_MAX);
		rtTrace(top_object, ray, prd);

		if (prd.done)
		{
			// We have hit the background or a luminaire
			prd.result += prd.radiance * prd.attenuation;
			break;
		}

		// Russian roulette termination 
		if (prd.depth >= rr_begin_depth)
		{
			float pcont = fmaxf(prd.attenuation);
			if (rnd(prd.seed) >= pcont)
				break;
			prd.attenuation /= pcont;
		}

		prd.depth++;
		prd.result += prd.radiance * prd.attenuation;

		// Update ray data for the next path segment
		ray_origin = prd.origin;
		ray_direction = prd.direction;
	}

	result += prd.result;
	seed = prd.seed;
}




