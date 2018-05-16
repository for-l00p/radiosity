#ifndef RADIOSITY_H
#define RADIOSITY_H

#include "Mesh.h"
#include "RadiosityFace.h"
#include "Ray.h"
#include <vector>
#include <iostream>
#include <chrono>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/glm.hpp>
#include<glm/gtx/vector_angle.hpp>
#include <glm/gtx/fast_square_root.hpp>
#include <glm/gtx/normal.hpp>

class Timer
{
public:
	Timer() : beg_(clock_::now()) {}
	void reset() { beg_ = clock_::now(); }
	double elapsed() const {
		return std::chrono::duration_cast<second_>
			(clock_::now() - beg_).count();
	}

private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	std::chrono::time_point<clock_> beg_;
};
using namespace std;
class Radiosity
{
public:
	void loadSceneFacesFromMesh(Mesh* mesh);
	void initEmittedEnergies();
	void initRadiosityValues();
	void calculateFormFactorsForFace(int i, int samplePoints);
	void PrepareUnshotRadiosityValues();
	void calculateRadiosityValues();
	glm::vec2 Radiosity::getTotalCounts(Mesh *mesh);
	bool doesRayHit(Ray* ray, int j, glm::vec3& hitPoint);
	bool isVisibleFrom(int i, int j);

	bool isVisibleFrom(glm::vec3 point_j, glm::vec3 point_i);

	bool isVisibleFrom(Ray input, int & global_k, float & global_distance, glm::vec3  & r_ij);

	bool isParallelToFace(Ray* r, int radFaceIndex);


	void setMeshFaceColors();

	int getMaxUnshotRadiosityFaceIndex();


private:

	vector<RadiosityFace> sceneFaces;
	vector<vector<double>> formFactors;

};

#endif