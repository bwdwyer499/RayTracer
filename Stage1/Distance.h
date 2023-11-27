#ifndef _DISTANCE_H
#define _DISTANCE_H

#include "Primitives.h"
#include "DFPrimitives.h"


// find distance to closest object
// apply boolean operators to distance calculations
inline float distance(const Point& currentPoint)
{
	Vector zero = { 0.0f, 0.0f, 0.0f };

	float min = FLT_MAX;
	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		float dist = 0;

		Vector distToCentrePoint = currentPoint - scene.objectContainer[i].sphere.pos;

		// call appropriate distance function based on the type of the object
		switch (scene.objectContainer[i].type)
		{
		case PrimitiveType::SPHERE:
			dist = (distToCentrePoint).length() - scene.objectContainer[i].sphere.size;
			break;
		case PrimitiveType::PLANE:
			dist = (scene.objectContainer[i].plane.normal) * (distToCentrePoint);
			break;
		case PrimitiveType::BOX:
		{
			Vector di = abs(distToCentrePoint) - scene.objectContainer[i].box.halfSize;

			dist = fminf(hmax(di), max(di, zero).length());
		}
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
	Vector zero = { 0.0f, 0.0f, 0.0f };

	DistanceAndIndex dai = { FLT_MAX, -1 };
	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		float dist = 0;

		Vector distToCentrePoint = currentPoint - scene.objectContainer[i].sphere.pos;

		// call appropriate distance function based on the type of the object
		switch (scene.objectContainer[i].type)
		{
		case PrimitiveType::SPHERE:
			dist = (distToCentrePoint).length() - scene.objectContainer[i].sphere.size;
			break;
		case PrimitiveType::PLANE:
			dist = (scene.objectContainer[i].plane.normal) * (distToCentrePoint);
			break;
		case PrimitiveType::BOX:
		{
			Vector di = abs(distToCentrePoint) - scene.objectContainer[i].box.halfSize;

			dist = fminf(hmax(di), max(di, zero).length());
		}
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
