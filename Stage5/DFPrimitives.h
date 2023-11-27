#ifndef _DF_PRIMITIVES_H
#define _DF_PRIMITIVES_H

#include "Primitives.h"
#include "PrimitivesSIMD.h"

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


inline __m256 boxDist(const __m256 posX, const __m256 posY, const __m256 posZ, const __m256 halfSizeX, const __m256 halfSizeY, const __m256 halfSizeZ, const Vector8& point)
{
	Vector8 boxPos(posX, posY, posZ);
	Vector8 boxHalfSize(halfSizeX, halfSizeY, halfSizeZ);

	Vector8 zero(0.0f, 0.0f, 0.0f);
	Vector8 di = abs(boxPos - point) - boxHalfSize;

	return _mm256_min_ps(hmax(di), max(di, zero).length());
}

inline __m256 sphereDist(const __m256 posX, const __m256 posY, const __m256 posZ, const __m256 size, const Vector8& point)
{
	Vector8 spherePos(posX, posY, posZ);

	return (point - spherePos).length() - size;
}


inline __m256 planeDist(const __m256 posX, const __m256 posY, const __m256 posZ, const __m256 normalX, const __m256 normalY, const __m256 normalZ, const Vector8& point)
{
	Vector8 planePos(posX, posY, posZ);
	Vector8 planeNormal(normalX, normalY, normalZ);

	return dot(planeNormal, point - planePos);
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