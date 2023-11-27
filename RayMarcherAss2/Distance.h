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

		// combine previous distance and current calculation depending on operator
		switch (scene.objectContainer[i].op)
		{
		case Operator::UNION:
			min = fminf(min, dist);
			break;
		case Operator::SUBTRACTION:
			min = fmaxf(min, -dist);
			break;
		case Operator::INTERSECTION:
			min = fmaxf(min, dist);
			break;
		case Operator::SMOOTH:
			{
				// this value should really be modifiable for each smooth
				// for this renderer, all the scene files will just have the appropriate size
				float k = 20.0f; 
				float h = clamp01(0.5f - 0.5f * (dist - min) / k);
				min = lerp(min, dist, h) - k * h * (1.0f - h);
			}
			//min = fminf(min, dist);
			break;
		}
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

		// combine previous distance and current calculation depending on operator
		switch (scene.objectContainer[i].op)
		{
		case Operator::UNION:
			dai = min(dai, dist, i);
			break;
		case Operator::SUBTRACTION:
			dai = max(dai, -dist, i);
			break;
		case Operator::INTERSECTION:
			dai = max(dai, dist, i);
			break;
		case Operator::SMOOTH:
			{
				float k = 20.0f;
				float h = clamp01(0.5f - 0.5f * (dist - dai.dist) / k);
				float min2 = lerp(dai.dist, dist, h) - k * h * (1.0f - h);
				// object that joins via smooth union, gets original object's material
				// in a perfect world we'd blend between them, but that would be a major new feature
				dai.dist = min2;
			}
			break;
		}
	}

	return dai;
}

#endif
