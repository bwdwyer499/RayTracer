#ifndef _DISTANCE_H
#define _DISTANCE_H

#include "Primitives.h"
#include "DFPrimitives.h"


// find distance to closest object
// apply boolean operators to distance calculations
inline __m256 distance(const Vector8& currentPoints)
{
	Vector8 vZeros(0.0f, 0.0f, 0.0f);
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 ones = _mm256_set1_ps(1.0f);

	__m256 finalDists = _mm256_set1_ps(FLT_MAX);

	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		Vector8 objectPoss(scene.objectContainer[i].plane.pos.x, scene.objectContainer[i].plane.pos.y, scene.objectContainer[i].plane.pos.z);
		Vector8 objectData(scene.objectContainer[i].plane.normal.x, scene.objectContainer[i].plane.normal.y, scene.objectContainer[i].plane.normal.z);

		Vector8 distToCentrePoint = currentPoints - objectPoss;

		__m256 dists = _mm256_setzero_ps();

		// distance to spheres, planes, and boxes
		__m256 sDist = (distToCentrePoint).length() - _mm256_set1_ps(scene.objectContainer[i].sphere.size);
		__m256 pDist = dot(objectData, distToCentrePoint);
		Vector8 di = abs(distToCentrePoint) - objectData;
		__m256 bDist = _mm256_min_ps(hmax(di), max(di, vZeros).length());

		__m256i types = _mm256_set1_epi32((int)scene.objectContainer[i].type);
		__m256 isSphere = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::SPHERE)));
		__m256 isPlane = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::PLANE)));
		__m256 isBox = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::BOX)));

		dists = _mm256_blendv_ps(dists, sDist, isSphere);
		dists = _mm256_blendv_ps(dists, pDist, isPlane);
		dists = _mm256_blendv_ps(dists, bDist, isBox);

		// combine previous distance and current calculation depending on operator
		switch (scene.objectContainer[i].op)
		{
		case Operator::UNION:
			__m256 cmp = _mm256_cmp_ps(dists, finalDists, _CMP_LT_OQ);
			finalDists = _mm256_blendv_ps(finalDists, dists, cmp);
			break;
		case Operator::SUBTRACTION:
			finalDists = _mm256_max_ps(finalDists, zeros - dists);
			break;
		case Operator::INTERSECTION:
			finalDists = _mm256_max_ps(finalDists, dists);
			break;
		case Operator::SMOOTH:
		{
			const __m256 half = _mm256_set1_ps(0.5f);
			const __m256 k = _mm256_set1_ps(20.0f);
			__m256 h = clamp01(half - half * (dists - finalDists) / k);
			__m256 min2 = lerp(finalDists, dists, h) - k * h * (ones - h);
			// object that joins via smooth union, gets original object's material
			// in a perfect world we'd blend between them, but that would be a major new feature
			finalDists = min2;
		}
		break;
		}
	}

	return finalDists;
}


// same as the previous function, but also return the index of the closest object
inline DistanceAndIndex8 distanceAndIndex(const Vector8& currentPoints)
{
	Vector8 vZeros(0.0f, 0.0f, 0.0f);
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 ones = _mm256_set1_ps(1.0f);

	DistanceAndIndex8 dai = { _mm256_set1_ps(FLT_MAX), _mm256_set1_epi32(-1) };

	for (int i = 0; i < (int)scene.numObjects; i++)
	{
		Vector8 objectPoss(scene.objectContainer[i].plane.pos.x, scene.objectContainer[i].plane.pos.y, scene.objectContainer[i].plane.pos.z);
		Vector8 objectData(scene.objectContainer[i].plane.normal.x, scene.objectContainer[i].plane.normal.y, scene.objectContainer[i].plane.normal.z);

		Vector8 distToCentrePoint = currentPoints - objectPoss;

		__m256 dists = _mm256_setzero_ps();

		// distance to spheres, planes, and boxes
		__m256 sDist = (distToCentrePoint).length() - _mm256_set1_ps(scene.objectContainer[i].sphere.size);
		__m256 pDist = dot(objectData, distToCentrePoint);
		Vector8 di = abs(distToCentrePoint) - objectData;
		__m256 bDist = _mm256_min_ps(hmax(di), max(di, vZeros).length());

		__m256i types = _mm256_set1_epi32((int)scene.objectContainer[i].type);
		__m256 isSphere = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::SPHERE)));
		__m256 isPlane = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::PLANE)));
		__m256 isBox = _mm256_castsi256_ps(_mm256_cmpeq_epi32(types, _mm256_set1_epi32((int)PrimitiveType::BOX)));

		dists = _mm256_blendv_ps(dists, sDist, isSphere);
		dists = _mm256_blendv_ps(dists, pDist, isPlane);
		dists = _mm256_blendv_ps(dists, bDist, isBox);

		__m256i indexes = _mm256_set1_epi32(i);
		__m256 cmp;

		// this was _much_ slower than just using the switch statement due to all the extra instructions done every iteration
		// particularly on scenes where the operation is the same every time (i.e. branch prediction works great)
		// (NOTE: some accuracy errors in this code)

		//__m256i ops = _mm256_set1_epi32((int) scene.objectContainer[i].op);
		//__m256i isUnion = _mm256_cmpeq_epi32(ops, _mm256_set1_epi32((int) Operator::UNION));
		//__m256i isSub = _mm256_cmpeq_epi32(ops, _mm256_set1_epi32((int)Operator::SUBTRACTION));
		//__m256i isInt = _mm256_cmpeq_epi32(ops, _mm256_set1_epi32((int)Operator::INTERSECTION));
		//__m256i isSmooth = _mm256_cmpeq_epi32(ops, _mm256_set1_epi32((int)Operator::SMOOTH));

		//__m256 cmpUnion = _mm256_cmp_ps(dists, dai.dist, _CMP_LT_OQ);

		//__m256 distUnion = _mm256_blendv_ps(dai.dist, dists, cmpUnion);
		//__m256i indexUnion = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmpUnion));

		//dai.dist = select(_mm256_castsi256_ps(isUnion), distUnion, dai.dist);
		//dai.index = select(isUnion, indexUnion, dai.index);

		//__m256 distSub = _mm256_max_ps(dai.dist, zeros - dists);
		//__m256 cmpSub = _mm256_cmp_ps(dai.dist, zeros - dists, _CMP_EQ_OQ);
		//__m256i indexSub = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmpSub));

		//dai.dist = select(_mm256_castsi256_ps(isSub), distSub, dai.dist);
		//dai.index = select(isSub, indexSub, dai.index);

		//__m256 distInt = _mm256_max_ps(dai.dist, dists);
		//__m256 cmpInt = _mm256_cmp_ps(dai.dist, dists, _CMP_EQ_OQ);
		//__m256i indexInt = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmpInt));

		//dai.dist = select(_mm256_castsi256_ps(isInt), distInt, dai.dist);
		//dai.index = select(isInt, indexInt, dai.index);

		//const __m256 half = _mm256_set1_ps(0.5f);
		//const __m256 k = _mm256_set1_ps(20.0f);
		//__m256 h = clamp01(half - half * (dists - dai.dist) / k);
		//__m256 distSmooth = lerp(dai.dist, dists, h) - k * h * (ones - h);

		//dai.dist = select(_mm256_castsi256_ps(isSmooth), distSmooth, dai.dist);

		// combine previous distance and current calculation depending on operator
		switch (scene.objectContainer[i].op)
		{
		case Operator::UNION:
			// due to problems with NaNs, this code causes deviations from the "perfect" images, and is replaced with something closer to the original code
			//min = _mm256_min_ps(dai.dist, dists);
			//cmp = _mm256_cmp_ps(dai.dist, dists, _CMP_EQ_OQ);
			//minIndex = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(minIndex), _mm256_castsi256_ps(indexes), cmp));
			cmp = _mm256_cmp_ps(dists, dai.dist, _CMP_LT_OQ);
			dai.dist = _mm256_blendv_ps(dai.dist, dists, cmp);
			dai.index = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmp));
			break;
		case Operator::SUBTRACTION:
			dai.dist = _mm256_max_ps(dai.dist, zeros - dists);

			cmp = _mm256_cmp_ps(dai.dist, zeros - dists, _CMP_EQ_OQ);
			dai.index = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmp));
			break;
		case Operator::INTERSECTION:
			dai.dist = _mm256_max_ps(dai.dist, dists);

			cmp = _mm256_cmp_ps(dai.dist, dists, _CMP_EQ_OQ);
			dai.index = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dai.index), _mm256_castsi256_ps(indexes), cmp));
			break;
		case Operator::SMOOTH:
			{
				const __m256 half = _mm256_set1_ps(0.5f);
				const __m256 k = _mm256_set1_ps(20.0f);
				__m256 h = clamp01(half - half * (dists - dai.dist) / k);
				__m256 min2 = lerp(dai.dist, dists, h) - k * h * (ones - h);
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