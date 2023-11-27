/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __SCENE_H
#define __SCENE_H

#include "SceneObjects.h"
#include <immintrin.h>

// description of a single static scene
typedef struct Scene 
{
	Point cameraPosition;					// camera location
	float cameraRotation;					// direction camera points
    float cameraFieldOfView;				// field of view for the camera

	float exposure;							// image exposure

	unsigned int skyboxMaterialId;

	// scene object counts
	unsigned int numMaterials;
	unsigned int numLights;
	unsigned int numObjects;

	// scene objects
	Material* materialContainer;	
	Light* lightContainer;
	SceneObject* objectContainer;

	// SoA materials
	float* reflection;

	// SIMD scene objects
	unsigned int numObjectsSIMD;
	__m256i* objectType;
	__m256* objectPosX;
	__m256* objectPosY;
	__m256* objectPosZ;
	union
	{
		struct
		{
			__m256* sphereSize;
		};
		struct
		{
			__m256* planeNormalX;
			__m256* planeNormalY;
			__m256* planeNormalZ;
		};
		struct
		{
			__m256* boxHalfSizeX;
			__m256* boxHalfSizeY;
			__m256* boxHalfSizeZ;
		};
	};
	__m256i* objectMaterialId;
	__m256i* objectOp;
} Scene;

// global scene file
extern Scene scene;

bool init(const char* inputName, Scene& scene);

#endif // __SCENE_H
