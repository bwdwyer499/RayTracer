#ifndef _DF_PRIMITIVES_H
#define _DF_PRIMITIVES_H

#include "Primitives.h"

// distance to a box from a point
inline float boxDist(const Box& box, const Point& point)
{
	Vector zero = { 0.0f, 0.0f, 0.0f };
	Vector di = abs(box.pos - point) - box.halfSize;

	return fminf(hmax(di), max(di, zero).length());
}

// distance to a sphere from a point
inline float sphereDist(const Sphere& sphere, const Point& point)
{
	return (point - sphere.pos).length() - sphere.size;
}

//distance to a plane from a point
inline float planeDist(const Plane& plane, const Point& point)
{
	return (plane.normal) * (point - plane.pos);
}


// structure combining a distance and an index and a couple of helper functions
struct DistanceAndIndex
{
	float dist;
	int index;
};

inline DistanceAndIndex min(DistanceAndIndex& current, float dist, int index)
{
	if (dist < current.dist)
	{
		DistanceAndIndex dai = { dist, index };
		return dai;
	}

	return current;
}

inline DistanceAndIndex max(DistanceAndIndex& current, float dist, int index)
{
	if (dist > current.dist)
	{
		DistanceAndIndex dai = { dist, index };
		return dai;
	}

	return current;
}

#endif