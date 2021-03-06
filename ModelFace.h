#ifndef MODELFACE_H
#define MODELFACE_H

#include "Material.h"

struct ModelFace
{
	vector<GLuint> vertexIndexes;
	vector<GLuint> textureIndexes;
	vector<GLuint> normalIndexes;
	Material* material;

	glm::vec3 intensity;
};

#endif