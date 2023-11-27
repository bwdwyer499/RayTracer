#ifndef _DISTANCE_H
#define _DISTANCE_H

#include "Primitives.h"
#include "DFPrimitives.h"
#include "PrimitivesSIMD.h"


// helper function to find "horizontal" minimum
__forceinline float selectMinimum(const __m256 values)
{
	// find min of elements 1&2, 3&4, 5&6, and 7&8
	__m256 minNeighbours = _mm256_min_ps(values, _mm256_permute_ps(values, 0x31));
	// find min of min(1,2)&min(5,6) and min(3,4)&min(7,8)
	__m256 minNeighbours2 = _mm256_min_ps(minNeighbours, _mm256_permute2f128_ps(minNeighbours, minNeighbours, 0x05));
	// find final minimum 
	__m256 mins = _mm256_min_ps(minNeighbours2, _mm256_permute_ps(minNeighbours2, 0x02));

	return mins.m256_f32[0];
}

// helper function to find "horizontal" minimum (and corresponding index value from another vector)
__forceinline DistanceAndIndex selectMinimumAndIndex(const __m256 values, const __m256i indexes)
{
	// find min of elements 1&2, 3&4, 5&6, and 7&8
	__m256 minNeighbours = _mm256_min_ps(values, _mm256_permute_ps(values, 0x31));
	// find min of min(1,2)&min(5,6) and min(3,4)&min(7,8)
	__m256 minNeighbours2 = _mm256_min_ps(minNeighbours, _mm256_permute2f128_ps(minNeighbours, minNeighbours, 0x05));
	// find final minimum 
	__m256 mins = _mm256_min_ps(minNeighbours2, _mm256_permute_ps(minNeighbours2, 0x02));

	// find all elements that match our minimum
	__m256i matchingTs = _mm256_castps_si256(_mm256_set1_ps(mins.m256_f32[0]) != values);
	// set all other elements to be MAX_INT (-1 but unsigned)
	__m256i matchingIndexes = matchingTs | indexes;

	// find minimum of remaining indexes (so smallest index will be chosen) using that same technique as above but with heaps of ugly casts
	__m256i minIndexNeighbours = _mm256_min_epu32(matchingIndexes, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(matchingIndexes), 0x31)));
	__m256i minIndexNeighbours2 = _mm256_min_epu32(minIndexNeighbours, _mm256_castps_si256(_mm256_permute2f128_ps(
		_mm256_castsi256_ps(minIndexNeighbours), _mm256_castsi256_ps(minIndexNeighbours), 0x05)));
	__m256i minIndex = _mm256_min_epu32(minIndexNeighbours2, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(minIndexNeighbours2), 0x02)));

	// "return" minimum and associated index through reference parameters
	return { mins.m256_f32[0], minIndex.m256i_i32[0] };
}


// find distance to closest object
inline float distance(const Point& currentPoint)
{
	Vector8 point(currentPoint.x, currentPoint.y, currentPoint.z);

	Vector8 zeros(0.0f, 0.0f, 0.0f);

	__m256 min = _mm256_set1_ps(FLT_MAX);
	for (int i = 0; i < (int)scene.numObjectsSIMD; i++)
	{
		Vector8 objectPoss(scene.objectPosX[i], scene.objectPosY[i], scene.objectPosZ[i]);
		Vector8 objectData(scene.planeNormalX[i], scene.planeNormalY[i], scene.planeNormalZ[i]);

		Vector8 distToCentrePoint = point - objectPoss;

		__m256 dists = _mm256_setzero_ps();

		// distance to spheres, planes, and boxes
		__m256 sDist = distToCentrePoint.length() - scene.sphereSize[i];
		__m256 pDist = dot(objectData, distToCentrePoint);
		Vector8 di = abs(distToCentrePoint) - objectData;
		__m256 bDist = _mm256_min_ps(hmax(di), max(di, zeros).length());

		// tests to see if objects are spheres, planes, or boxes
		__m256 isSphere = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::SPHERE)));
		__m256 isPlane = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::PLANE)));
		__m256 isBox = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::BOX)));

		// choose correct distance based on object type
		dists = _mm256_blendv_ps(dists, sDist, isSphere);
		dists = _mm256_blendv_ps(dists, pDist, isPlane);
		dists = _mm256_blendv_ps(dists, bDist, isBox);

		// find minimum (per SIMD column)
		min = _mm256_min_ps(min, dists);
	}

	return selectMinimum(min);
}

// same as the previous function, but also return the index of the closest object
inline DistanceAndIndex distanceAndIndex(const Point& currentPoint)
{
	Vector8 point(currentPoint.x, currentPoint.y, currentPoint.z);

	Vector8 zeros(0.0f, 0.0f, 0.0f);

	__m256 min = _mm256_set1_ps(FLT_MAX);
	__m256i minIndex = _mm256_set1_epi32(-1);
	__m256i indexes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	for (unsigned int i = 0; i < scene.numObjectsSIMD; i++)
	{
		Vector8 objectPoss(scene.objectPosX[i], scene.objectPosY[i], scene.objectPosZ[i]);
		Vector8 objectData(scene.planeNormalX[i], scene.planeNormalY[i], scene.planeNormalZ[i]);

		Vector8 distToCentrePoint = point - objectPoss;

		__m256 dists = _mm256_setzero_ps();

		// distance to spheres, planes, and boxes
		__m256 sDist = (distToCentrePoint).length() - scene.sphereSize[i];
		__m256 pDist = dot(objectData, distToCentrePoint);
		Vector8 di = abs(distToCentrePoint) - objectData;
		__m256 bDist = _mm256_min_ps(hmax(di), max(di, zeros).length());

		__m256 isSphere = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::SPHERE)));
		__m256 isPlane = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::PLANE)));
		__m256 isBox = _mm256_castsi256_ps(_mm256_cmpeq_epi32(scene.objectType[i], _mm256_set1_epi32((int)PrimitiveType::BOX)));

		dists = _mm256_blendv_ps(dists, sDist, isSphere);
		dists = _mm256_blendv_ps(dists, pDist, isPlane);
		dists = _mm256_blendv_ps(dists, bDist, isBox);

		min = _mm256_min_ps(min, dists);

		__m256 cmp = _mm256_cmp_ps(min, dists, _CMP_EQ_OQ);
		minIndex = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(minIndex), _mm256_castsi256_ps(indexes), cmp));

		indexes = indexes + _mm256_set1_epi32(8);
	}

	return selectMinimumAndIndex(min, minIndex);
}

#endif
