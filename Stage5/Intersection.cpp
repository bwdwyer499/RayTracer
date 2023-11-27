/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#include "Intersection.h"
#include "DFPrimitives.h"
#include "Lighting.h"
#include "Distance.h"
#include "PrimitivesSIMD.h"


Vector8 calculateNormal(const Vector8& p)
{
	const __m256 EPSILONS = _mm256_set1_ps(EPSILON);

	Vector8 n(distance(Vector8(p.xs + EPSILONS, p.ys, p.zs)) - distance(Vector8(p.xs - EPSILONS, p.ys, p.zs)),
		distance(Vector8(p.xs, p.ys + EPSILONS, p.zs)) - distance(Vector8(p.xs, p.ys - EPSILONS, p.zs)),
		distance(Vector8(p.xs, p.ys, p.zs + EPSILONS)) - distance(Vector8(p.xs, p.ys, p.zs - EPSILONS)));

	return normalise(n);
}

void calculateIntersectionResponse_SIMD(const Scene& scene, const Vector8& viewRayPos, const Vector8& viewRayDir, __m256i index, Intersection8& intersects)
{
	intersects.normal = calculateNormal(intersects.pos);
	intersects.materialIndex = _mm256_i32gather_epi32((const int*)scene.objectMaterialId, index, 4);

	// calculate view projection
	intersects.viewProjection = dot(viewRayDir, intersects.normal);
}

// find intersection with object along a ray
Vector8 marchRay(Vector8 viewRayPos, Vector8 viewRayDir, __m256i& index)
{
	Vector8 currentPoints = viewRayPos;

	__m256 distance = _mm256_set1_ps(FLT_MAX);
	__m256i indexes = _mm256_set1_epi32(-1);

	__m256 ones = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF));
	__m256 done = _mm256_castsi256_ps(_mm256_set1_epi32(0));				//TODO: ??? reverse this logic?

	const __m256 EPSILONS = _mm256_set1_ps(EPSILON);
	const __m256 MIN_MARCH_STEPS = _mm256_set1_ps(MIN_MARCH_STEP);

	DistanceAndIndex8 dai;
	dai.dist = _mm256_set1_ps(FLT_MAX);
	dai.index = _mm256_set1_epi32(-1);

	// progress a little along the ray to make sure we aren't too close to the previous object
	currentPoints = currentPoints + viewRayDir * EPSILONS;

	// loop until too many steps taken or collided with object (for all rays)
	for (int steps = 0; steps < MAX_MARCH_STEPS; ++steps)
	{
		DistanceAndIndex8 daiTemp = distanceAndIndex(currentPoints);

		dai.dist = select(!done, daiTemp.dist, dai.dist);
		dai.index = select(_mm256_castps_si256(!done), daiTemp.index, dai.index);

		// if we are very close to something, consider that an intersection and mark this SIMD lane as done
		__m256 smallerThanEPSILON = _mm256_cmp_ps(dai.dist, EPSILONS, _CMP_LT_OQ);
		Vector8 newPos1 = currentPoints + viewRayDir * (dai.dist - EPSILONS);
		Vector8 newPos2 = currentPoints + viewRayDir * dai.dist;
		currentPoints = select(smallerThanEPSILON & !done, newPos1, currentPoints);
		done = done | smallerThanEPSILON;

		// make sure than step size isn't too small
		__m256 smallerThanMinStep = _mm256_cmp_ps(dai.dist, MIN_MARCH_STEPS, _CMP_LT_OQ);
		dai.dist = select(smallerThanMinStep & !done, MIN_MARCH_STEPS, dai.dist);

		if (_mm256_test_all_ones(_mm256_castps_si256(done))) break;

		// move the point
		currentPoints = select(!done, newPos2, currentPoints);
	}

	// return the index of the closest object (for all rays)
	index = dai.index;

	return currentPoints;
}

Colour8 traceRay(const Scene& scene, Vector8 viewRayPos, Vector8 viewRayDir)
{
	Colour8 output = { 0 };
	__m256 coef = _mm256_set1_ps(1.0f);								// amount of ray left to transmit
	__m256i done = _mm256_setzero_si256();							// whether each SIMD lane is finished calculation or not
	Intersection8 intersects;										// properties of current intersection (8 rays needs 8 intersections)

	// loop until reached maximum ray cast limit (unless loop is broken out of)
	for (int level = 0; level < MAX_RAYS_CAST; ++level)
	{
		// check for intersections between the view ray and any of the objects in the scene
		__m256i index = _mm256_set1_epi32(-1);

		Vector8 pos = marchRay(viewRayPos, viewRayDir, index);

		// keep only the positions for SIMD lanes that aren't done
		intersects.pos = select(!done, pos, intersects.pos);

		// mark this SIMD lane as done (i.e. "exit" the loop) if no intersection found
		done = done | _mm256_cmpgt_epi32(_mm256_setzero_si256(), index);

		// calculate response to collision: ie. get normal at point of collision and material of object
		calculateIntersectionResponse_SIMD(scene, viewRayPos, viewRayDir, index, intersects);

		// ensure completed rays don't cause array-out-of-bounds issues
		intersects.materialIndex = select(done, _mm256_setzero_si256(), intersects.materialIndex);

		output = select(done, output, output + coef * applyLighting(scene, viewRayPos, viewRayDir, intersects));

		__m256 reflections = _mm256_i32gather_ps(scene.reflection, intersects.materialIndex, 4); // look up 8 indexes at once
		__m256 reflectionsIsZero = _mm256_cmp_ps(reflections, _mm256_setzero_ps(), _CMP_EQ_OQ);

		Vector8 reflectedDir = viewRayDir - intersects.normal * intersects.viewProjection * _mm256_set1_ps(2.0f);

		// if object has reflection component, adjust the view ray and coefficent of calculation and continue looping
		__m256 notDoneAndReflecty = _mm256_castsi256_ps(!done) & (!reflectionsIsZero);
		viewRayPos = select(notDoneAndReflecty, intersects.pos, viewRayPos);
		viewRayDir = select(notDoneAndReflecty, reflectedDir, viewRayDir);
		coef = select(notDoneAndReflecty, coef * reflections, coef);

		// if no reflection then finish looping (cast no more rays)
		__m256 notDoneAndNotReflecty = _mm256_castsi256_ps(!done) & (reflectionsIsZero);
		coef = select(notDoneAndNotReflecty, _mm256_setzero_ps(), coef);
		done = done | _mm256_castps_si256(reflectionsIsZero);
	}

	// if the calculation coefficient is non-zero, read from the environment map
	__m256 sampleSkybox = _mm256_cmp_ps(coef, _mm256_setzero_ps(), _CMP_GT_OQ);
	Colour skyboxColour = scene.materialContainer[scene.skyboxMaterialId].diffuse;

	output.reds = select(sampleSkybox, output.reds + coef * _mm256_set1_ps(skyboxColour.red), output.reds);
	output.greens = select(sampleSkybox, output.greens + coef * _mm256_set1_ps(skyboxColour.green), output.greens);
	output.blues = select(sampleSkybox, output.blues + coef * _mm256_set1_ps(skyboxColour.blue), output.blues);

	return output;
}