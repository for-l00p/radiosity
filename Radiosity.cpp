#include "Radiosity.h"
#include <glm/gtx/intersect.hpp>
// helper functions and utilities to work with CUDA

#include <math.h>



#define INITIAL_LIGHT_EMITTER_INTENSITY		15.0f
#define INITIAL_AMBIENT_INTENSITY			glm::vec3(0.00f, 0.00f, 0.00f)

#define RADIOSITY_SOLUTION_THRESHOLD		glm::vec3(0.25f, 0.25f, 0.25f)
#define FORM_FACTOR_SAMPLES					750


void Radiosity::loadSceneFacesFromMesh(Mesh* mesh)
{
	sceneFaces.clear();
	formFactors.clear();

	for(int i=0; i<mesh->sceneModel.size(); i++)
	{
		//for every scene object
		ObjectModel* currentObject = &mesh->sceneModel[i].obj_model;

		int currentObjFaces = currentObject->faces.size();

		for(int j=0; j < currentObjFaces; j++)
		{
			//for every face of it
			ModelFace* currentFace = &mesh->sceneModel[i].obj_model.faces[j];

			RadiosityFace radiosityFace;

			radiosityFace.model = currentObject;
			radiosityFace.faceIndex = j;
			if(currentFace->material->illuminationMode != 1)
				radiosityFace.emission = currentFace->material->diffuseColor * INITIAL_LIGHT_EMITTER_INTENSITY;
			else
				radiosityFace.emission = currentFace->material->diffuseColor * INITIAL_AMBIENT_INTENSITY;

			radiosityFace.totalRadiosity = radiosityFace.emission;
			radiosityFace.unshotRadiosity = radiosityFace.emission;

			sceneFaces.push_back(radiosityFace);
		}
	}

	formFactors.resize(sceneFaces.size(), vector<double>(sceneFaces.size()));
}

int Radiosity::getMaxUnshotRadiosityFaceIndex()
{
	int index = -1;
	double maxUnshot = 0.0;

	for(int i=0; i<sceneFaces.size(); i++)
	{
		double curUnshot = glm::length(sceneFaces[i].unshotRadiosity);
		double curArea = sceneFaces[i].model->getFaceArea(sceneFaces[i].faceIndex);

		curUnshot *= curArea;

		if(curUnshot > maxUnshot)
		{
			index = i;
			maxUnshot = curUnshot;
		}
	}
	return index;

}


glm::vec3 getCosineDistributionVector(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 norm) {
	float sin_theta = sqrt((double)rand() / ((double)RAND_MAX + 1));
	float cos_theta = sqrt(1 - sin_theta * sin_theta);

	float psi = ((double)rand() / ((double)RAND_MAX + 1) * 2 * 3.14159265359);

	glm::vec3 a1 = glm::vec3(sin_theta) * cos(psi);
	glm::vec3 b1 = glm::vec3(sin_theta) * sin(psi);
	glm::vec3 c1 = glm::vec3(cos_theta);


	glm::vec3 v1 = a1 * (a - b);
	glm::vec3 v2 = b1 * (a - c);
	glm::vec3 v3 = c1 * norm;

	return v1 + v2 + v3;
}

void Radiosity::calculateFormFactorsForFace(int i, int samplePointsCount)
{	

	/*	
	// input data
	int len = sceneFaces.size();
	// the data has some zero padding at the end so that the size is a multiple of
	// four, this simplifies the processing as each thread can process four
	// elements (which is necessary to avoid bank conflicts) but no branching is
	// necessary to avoid out of bounds reads

	vector <float> area (len);
	vector <glm::vec3> normal(len);
	vector <glm::vec3> verticies();

	for (int i = 0; i < len; i++)
	{
		area[i] = (sceneFaces[i].model->getFaceArea(sceneFaces[i].faceIndex));
		normal[i] = (sceneFaces[i].model->getFaceNormal(sceneFaces[i].faceIndex));

	}

;

	// Use int2 showing that CUDA vector types can be used in cpp code
	float1 area2[1000000];
	float3 normal[1000000];
	float3 verticies[1000000];

	
	for (int i = 0; i < len; i++)
	{
		area2[i].x = area[i];
		normal[i].x = normal[i].x;
		normal[i].y = normal[i].y;
		normal[i].z = normal[i].z;

		

	}

	bool bTestResult;

	// run the device part of the program
	bTestResult = runTest(0, NULL, str, i2, len);

	std::cout << str << std::endl;

	char str_device[16];

	for (int i = 0; i < len; i++)
	{
		str_device[i] = (char)(i2[i].x);
	}

	std::cout << str_device << std::endl;

	*/

	printf("Calculating form factors for face %d with %d sample points per form factor\n", i, samplePointsCount);
	

	vector<Ray> generated_dir(samplePointsCount);

	vector<glm::vec3> samplePoints_i = sceneFaces[i].model->monteCarloSamplePoints(sceneFaces[i].faceIndex, samplePointsCount);

	glm::vec3 normal_i = sceneFaces[i].model->getFaceNormal(sceneFaces[i].faceIndex);


	int v0_k_index = sceneFaces[i].model->faces[sceneFaces[i].faceIndex].vertexIndexes[0];
	int v1_k_index = sceneFaces[i].model->faces[sceneFaces[i].faceIndex].vertexIndexes[1];
	int v2_k_index = sceneFaces[i].model->faces[sceneFaces[i].faceIndex].vertexIndexes[2];

	glm::vec3 A = sceneFaces[i].model->vertices[v0_k_index];
	glm::vec3 B = sceneFaces[i].model->vertices[v1_k_index];
	glm::vec3 C = sceneFaces[i].model->vertices[v2_k_index];


	//We generate a ray and direction based on the various different locations on the face
	for (int j = 0; j < samplePointsCount; j++) {

		glm::vec3 direction = getCosineDistributionVector(A,B,C,normal_i);

		generated_dir[j] = Ray(samplePoints_i[j], glm::normalize(direction));

	}
	

	for (int j = 0; j < samplePointsCount; j++) {
		int k;
		float  distance; 
		glm::vec3 HitPoint;
		if (isVisibleFrom(generated_dir[j], k, distance, HitPoint)) {
			//printf("The hitpoint is %f %f %f \n", HitPoint[0], HitPoint[1], HitPoint[2]);
			float area_i = sceneFaces[i].model->getFaceArea(sceneFaces[i].faceIndex);
			glm::vec3 normal_j = sceneFaces[k].model->getFaceNormal(sceneFaces[k].faceIndex);

			glm::vec3 r_ij = glm::normalize(HitPoint - samplePoints_i[j]);

			//printf("The ray is %f %f %f \n", r_ij[0], r_ij[1], r_ij[2]);

			float r_squared = distance * distance;

			double cos_angle_i = glm::dot(r_ij, normal_i);
			double cos_angle_j = glm::dot(r_ij, normal_j);

			double delta_F = (cos_angle_i * cos_angle_j) / (3.14159265359 * r_squared + (area_i));

			//printf("The delta_F is %f \n", delta_F);

			if (abs(delta_F) > 0.0) 
				formFactors[i][k] = formFactors[i][k] + abs(delta_F);
			//printf("The F for %d %d is %f \n \n", i, k, formFactors[i][k]);

		}

	}

}

void Radiosity::PrepareUnshotRadiosityValues()
{
	for(int i=0; i<sceneFaces.size(); i++)
	{
		sceneFaces[i].totalRadiosity = sceneFaces[i].emission;
		sceneFaces[i].unshotRadiosity = sceneFaces[i].emission;
	}
}

void Radiosity::calculateRadiosityValues()
{
	glm::vec3 threshold = RADIOSITY_SOLUTION_THRESHOLD;
	glm::vec3 error(100.0f, 100.0f, 100.0f);

	/*
	for(int i=0; i<sceneFaces.size(); i++)
	{
		sceneFaces[i].totalRadiosity = sceneFaces[i].emission;
		sceneFaces[i].unshotRadiosity = sceneFaces[i].emission;
	}
	*/
	int iterations = 0;
	 while (
			error.r > threshold.r &&
			error.g > threshold.g &&
			error.b > threshold.b && iterations < 100
		)
	 {
		int i = getMaxUnshotRadiosityFaceIndex();

		calculateFormFactorsForFace(i, FORM_FACTOR_SAMPLES);

		 for (int j=0; j< sceneFaces.size(); j++)
		 {
			glm::dvec3 p_j = (glm::dvec3)sceneFaces[j].model->faces[sceneFaces[j].faceIndex].material->diffuseColor;

			glm::dvec3 delta_rad = sceneFaces[i].unshotRadiosity * formFactors[i][j] * p_j;

			sceneFaces[j].unshotRadiosity = sceneFaces[j].unshotRadiosity + delta_rad;
			sceneFaces[j].totalRadiosity  = sceneFaces[j].totalRadiosity + delta_rad;
		 }

		 sceneFaces[i].unshotRadiosity = glm::dvec3(0.0f, 0.0f, 0.0f);

		int e = getMaxUnshotRadiosityFaceIndex();
		error = sceneFaces[e].unshotRadiosity;
		iterations = iterations + 1;
	 }
	
}

void Radiosity::setMeshFaceColors()
{
	for (int i=0; i< sceneFaces.size(); i++)
	{
		glm::dvec3 radValue = glm::clamp(sceneFaces[i].totalRadiosity,0.0,1.0);

		glm::clamp(radValue,0.0,1.0);
		sceneFaces[i].model->faces[sceneFaces[i].faceIndex].intensity = radValue;
	}
}

bool Radiosity::isParallelToFace(Ray* r, int i)
{
	glm::vec3 n = sceneFaces[i].model->getFaceNormal(sceneFaces[i].faceIndex);
	float dotProduct = abs(glm::dot(n, r->getDirection()));
	if(dotProduct <= 0.001f)
		return true;
	return false;
}

struct RayHit
{
	float distance;
	int hitSceneFaceIndex;	
	glm::vec3 hitpoint;
};

bool rayHit_LessThan(RayHit r1, RayHit r2)
{
	if(r1.distance < r2.distance)
		return true;
	return false;
}

bool Radiosity::isVisibleFrom(int i, int j)
{

	vector<RayHit> rayHits;

	//get both centroids
	glm::vec3 centroid_i = sceneFaces[i].model->getFaceCentroid(sceneFaces[i].faceIndex);
	glm::vec3 centroid_j = sceneFaces[j].model->getFaceCentroid(sceneFaces[j].faceIndex);

	//now make a ray
	Ray ray(centroid_i, centroid_j - centroid_i);

	for(int k=0; k<sceneFaces.size(); k++)
	{
		if(k == i)
			continue;

		int v0_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[0];
		int v1_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[1];
		int v2_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[2];

		glm::vec3 A = sceneFaces[k].model->vertices[v0_k_index];
		glm::vec3 B = sceneFaces[k].model->vertices[v1_k_index];
		glm::vec3 C = sceneFaces[k].model->vertices[v2_k_index];

		glm::vec3 hitPoint;
		if(glm::intersectRayTriangle(centroid_i, centroid_j - centroid_i, A, B, C, hitPoint))
		{
			RayHit currentHit;
			currentHit.distance = glm::distance(ray.getStart(), hitPoint);
			currentHit.hitSceneFaceIndex = k;
			rayHits.push_back(currentHit);
		}
		/*
		if(doesRayHit(&ray, k, hitPoint))
		{
			RayHit currentHit;
			currentHit.distance = glm::distance(ray.getStart(), hitPoint);
			currentHit.hitSceneFaceIndex = k;
			rayHits.push_back(currentHit);
		}
		*/
	}
	if(rayHits.empty())
		return false;

	if(rayHits.size() == 1)
		return true;

	return false;
}

bool Radiosity::isVisibleFrom(glm::vec3 point_j, glm::vec3 point_i)
{
	vector<RayHit> rayHits;

	//now make a ray
	Ray ray(point_i, point_j - point_i);

	for(int k=0; k<sceneFaces.size(); k++)
	{
		int v0_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[0];
		int v1_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[1];
		int v2_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[2];

		glm::vec3 A = sceneFaces[k].model->vertices[v0_k_index];
		glm::vec3 B = sceneFaces[k].model->vertices[v1_k_index];
		glm::vec3 C = sceneFaces[k].model->vertices[v2_k_index];

		glm::vec3 hitPoint;
		if(glm::intersectRayTriangle(point_i, point_j - point_i, A, B, C, hitPoint))
		{
			RayHit currentHit;
			currentHit.distance = glm::distance(ray.getStart(), hitPoint);
			currentHit.hitSceneFaceIndex = k;
			rayHits.push_back(currentHit);
		}
		
	}
	
	// return's the values if it hits
	return (rayHits.size() == 1);
		
}

bool Radiosity::isVisibleFrom(Ray input, int & global_k, float & global_distance, glm::vec3  & r_ij)
{
	vector<RayHit> rayHits;
	Ray ray = input;

	for (int k = 0; k<sceneFaces.size(); k++)
	{
		int v0_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[0];
		int v1_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[1];
		int v2_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[2];

		glm::vec3 A = sceneFaces[k].model->vertices[v0_k_index];
		glm::vec3 B = sceneFaces[k].model->vertices[v1_k_index];
		glm::vec3 C = sceneFaces[k].model->vertices[v2_k_index];

		glm::vec3 hitPoint;
		RayHit currentHit;
		if (sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes.size() > 3) {
			int v3_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[3];
			glm::vec3 D = sceneFaces[k].model->vertices[v3_k_index];

			if (glm::intersectRayTriangle(input.getStart(), input.getDirection(), A, B, D, hitPoint)) {
				currentHit.distance = glm::distance(ray.getStart(), hitPoint);
				currentHit.hitpoint = hitPoint;
				currentHit.hitSceneFaceIndex = k;
				hitPoint = glm::vec3(0.0f);
			}
			else if (glm::intersectRayTriangle(input.getStart(), input.getDirection(), C, B, D, hitPoint)) {
				currentHit.distance = glm::distance(ray.getStart(), hitPoint);
				currentHit.hitpoint = hitPoint;
				currentHit.hitSceneFaceIndex = k;
				rayHits.push_back(currentHit);
				hitPoint = glm::vec3(0.0f);
			}
			else if (glm::intersectRayTriangle(input.getStart(), input.getDirection(), A, C, D, hitPoint)) {
				currentHit.distance = glm::distance(ray.getStart(), hitPoint);
				currentHit.hitpoint = hitPoint;
				currentHit.hitSceneFaceIndex = k;
				rayHits.push_back(currentHit);
				hitPoint = glm::vec3(0.0f);
			}
			else if (glm::intersectRayTriangle(input.getStart(), input.getDirection(), A, B, C, hitPoint)){
				currentHit.distance = glm::distance(ray.getStart(), hitPoint);
				currentHit.hitpoint = hitPoint;
				currentHit.hitSceneFaceIndex = k;
				rayHits.push_back(currentHit);
				hitPoint = glm::vec3(0.0f);
			}

			hitPoint = glm::vec3(0.0f);
		}
		else if (glm::intersectRayTriangle(input.getStart(), input.getDirection(), A, B, C, hitPoint)){
			currentHit.distance = glm::distance(ray.getStart(), hitPoint);
			currentHit.hitpoint = hitPoint;
			currentHit.hitSceneFaceIndex = k;
			rayHits.push_back(currentHit);
		}
		

	}
	
	if (rayHits.size() >= 1){
		global_k = rayHits[0].hitSceneFaceIndex;
		float distance = rayHits[0].distance;
		global_distance = distance;
		r_ij = rayHits[0].hitpoint;

		for (int j = 1; j < rayHits.size(); j++) {

			if (rayHits[j].distance < distance) {
				global_k = rayHits[j].hitSceneFaceIndex;
				distance = rayHits[j].distance;
				global_distance = distance;
				r_ij = rayHits[j].hitpoint;

			}
		}

	}

	// return's the values if it hits
	return (rayHits.size() == 1);

}



bool Radiosity::doesRayHit(Ray* ray, int k, glm::vec3& hitPoint)
{
	//check if ray is parallel to face
	glm::vec3 n_k = sceneFaces[k].model->getFaceNormal(sceneFaces[k].faceIndex);

	//if(isParallelToFace(ray, k))
	//	return false;

	//get all vertices of k
	int v0_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[0];
	int v1_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[1];
	int v2_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[2];

	glm::vec3 A = sceneFaces[k].model->vertices[v0_k_index];
	glm::vec3 B = sceneFaces[k].model->vertices[v1_k_index];
	glm::vec3 C = sceneFaces[k].model->vertices[v2_k_index];
	
	//first we handle the case where patch k has only 3 vertices

	//plane equasion
	//ax + by + cz = d
	//n.x=d
	float a = n_k.x;
	float b = n_k.y;
	float c = n_k.z;
	float d = glm::dot(n_k, A);
	
	//ray equasion
	//R(t) = ray.start + t*ray.direction
	//plug into plane equasion, solve for t

	float t = (d - glm::dot(n_k, ray->getStart())) / glm::dot(n_k, ray->getDirection());

	//triangle behind ray
	if(t<0.0f)
		return false;

	//calculate the intersection point between the ray and the plane where k lies
	glm::vec3 Q = ray->getStart() + (t * ray->getDirection());

	//now we determine if Q is inside or outside of k
	if(sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes.size() == 3)
	{
		float AB_EDGE = glm::dot(glm::cross((B - A), (Q - A)) , n_k);
		float BC_EDGE = glm::dot(glm::cross((C - B), (Q - B)) , n_k);
		float CA_EDGE = glm::dot(glm::cross((A - C), (Q - C)) , n_k);

		if(AB_EDGE >= 0.0f && BC_EDGE >= 0.0f && CA_EDGE >= 0.0f)
		{
			float ABC_AREA_DOUBLE = glm::dot(glm::cross((B - A), (C - A)) , n_k);

			float alpha = BC_EDGE / ABC_AREA_DOUBLE;
			float beta  = CA_EDGE / ABC_AREA_DOUBLE;
			float gamma = AB_EDGE / ABC_AREA_DOUBLE;

			hitPoint = glm::vec3(alpha * A + beta * B + gamma * C);
			return true;
		}
		else
			return false;		
	}
	else
	{
		int v3_k_index = sceneFaces[k].model->faces[sceneFaces[k].faceIndex].vertexIndexes[3];
		glm::vec3 D = sceneFaces[k].model->vertices[v3_k_index];

		float AB_EDGE = glm::dot(glm::cross((B - A), (Q - A)) , n_k);
		float BD_EDGE = glm::dot(glm::cross((D - B), (Q - B)) , n_k);
		float DA_EDGE = glm::dot(glm::cross((A - D), (Q - D)) , n_k);

		float BC_EDGE = glm::dot(glm::cross((C - B), (Q - B)) , n_k);
		float CD_EDGE = glm::dot(glm::cross((D - C), (Q - C)) , n_k);
		float DB_EDGE = glm::dot(glm::cross((B - D), (Q - D)) , n_k);
		
		if(AB_EDGE >= 0.0f && BD_EDGE >= 0.0f && DA_EDGE >= 0.0f)
		{
			float ABD_AREA_DOUBLE = glm::dot(glm::cross((B - A), (D - A)) , n_k);

			float alpha = BD_EDGE / ABD_AREA_DOUBLE;
			float beta  = DA_EDGE / ABD_AREA_DOUBLE;
			float gamma = AB_EDGE / ABD_AREA_DOUBLE;

			hitPoint = glm::vec3(alpha * A + beta * B + gamma * D);
			return true;
		}
		else if(BC_EDGE >= 0.0f && CD_EDGE >= 0.0f && DB_EDGE >= 0.0f)
		{
			float BCD_AREA_DOUBLE = glm::dot(glm::cross((B - C), (D - C)) , n_k);

			float alpha = CD_EDGE / BCD_AREA_DOUBLE;
			float beta  = DB_EDGE / BCD_AREA_DOUBLE;
			float gamma = BC_EDGE / BCD_AREA_DOUBLE;

			hitPoint = glm::vec3(alpha * B + beta * C + gamma * D);
			return true;
		}
		else
			return false;
	}
	return false;
}