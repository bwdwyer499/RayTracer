#ifndef _DISTANCE_H
#define _DISTANCE_H

#include "Primitives.h"
#include "DFPrimitives.h"


// find distance to closest object
// apply boolean operators to distance calculations
inline float distance(const Point& currentPoint)
{
	float min = FLT_MAX;
	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		float dist = 0;

		// call appropriate distance function based on the type of the object
		switch (scene.objectContainer[i].type)
		{
		case PrimitiveType::SPHERE:
			dist = sphereDist(scene.objectContainer[i].sphere, currentPoint);
			break;
		case PrimitiveType::PLANE:
			dist = planeDist(scene.objectContainer[i].plane, currentPoint);
			break;
		case PrimitiveType::BOX:
			dist = boxDist(scene.objectContainer[i].box, currentPoint);
			break;
		case PrimitiveType::NONE:
			dist = 0;
			break;
		}

		// combine previous distance and current calculation using a union operator (i.e. min)
		// NOTE: this is a more simplified and limited approach than the code in Assignment 1
		// NOTE: this means that not all Assignment 1 scenes will render correctly
		min = fminf(min, dist);
	}

	return min;
}


// same as the previous function, but also return the index of the closest object
inline DistanceAndIndex distanceAndIndex(const Point& currentPoint)
{
	DistanceAndIndex dai = { FLT_MAX, -1 };

	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		float dist = 0;

		// call appropriate distance function based on the type of the object
		switch (scene.objectContainer[i].type)
		{
		case PrimitiveType::SPHERE:
			dist = sphereDist(scene.objectContainer[i].sphere, currentPoint);
			break;
		case PrimitiveType::PLANE:
			dist = planeDist(scene.objectContainer[i].plane, currentPoint);
			break;
		case PrimitiveType::BOX:
			dist = boxDist(scene.objectContainer[i].box, currentPoint);
			break;
		case PrimitiveType::NONE:
			dist = 0;
			break;
		}

		// combine previous distance and current calculation using a union operator (i.e. min)
		// NOTE: this is a more simplified and limited approach than the code in Assignment 1
		// NOTE: this means that not all Assignment 1 scenes will render correctly
		dai = min(dai, dist, i);
	}

	return dai;
}

#endif
