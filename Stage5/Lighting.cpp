/*  The following code is a VERY heavily modified from code originally sourced from:
Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#include "Lighting.h"
#include "Colour.h"
#include "Intersection.h"
#include "Texturing.h"
#include "Distance.h"


Colour8 applyCheckerboard(const Intersection8& intersects, const Colour8& diffuse, const Colour8& diffuse2, const Vector8& offset, const __m256& size)
{
	Vector8 p = (intersects.pos - offset) / size;

	__m256i which = (_mm256_cvtps_epi32(_mm256_floor_ps(p.xs) + _mm256_floor_ps(p.ys) + _mm256_floor_ps(p.zs)) & _mm256_set1_epi32(1));

	return select(_mm256_mullo_epi32(which, _mm256_set1_epi32(0xFFFFFFFF)), diffuse, diffuse2);
}


// apply computed circular texture
Colour8 applyCircles(const Intersection8& intersects, const Colour8& diffuse, const Colour8& diffuse2, const Vector8& offset, const __m256& size)
{
	Vector8 p = (intersects.pos - offset) / size;

	__m256i which = (_mm256_cvtps_epi32(_mm256_floor_ps(_mm256_sqrt_ps(p.xs * p.xs + p.ys * p.ys + p.zs * p.zs))) & _mm256_set1_epi32(1));

	return select(_mm256_mullo_epi32(which, _mm256_set1_epi32(0xFFFFFFFF)), diffuse, diffuse2);
}


//// apply computed wood grain texture
Colour8 applyWood(const Intersection8& intersects, const Colour8& diffuse, const Colour8& diffuse2, const Vector8& offset, const __m256& size)
{
	Vector8 p = (intersects.pos - offset) / size;

	// squiggle up where the point is
	p = { 
		p.xs * _mm256_cos_ps(p.ys * _mm256_set1_ps(0.666f)) * _mm256_sin_ps(p.zs * _mm256_set1_ps(1.027f)), 
		_mm256_cos_ps(p.xs) * p.ys * _mm256_sin_ps(p.zs * _mm256_set1_ps(1.212f)),
		_mm256_cos_ps(p.xs * _mm256_set1_ps(1.471f)) * _mm256_cos_ps(p.ys * _mm256_set1_ps(0.793f)) * p.zs 
	};

	__m256i which = (_mm256_cvtps_epi32(_mm256_floor_ps(_mm256_sqrt_ps(p.xs * p.xs + p.ys * p.ys + p.zs * p.zs))) & _mm256_set1_epi32(1));

	return select(_mm256_mullo_epi32(which, _mm256_set1_epi32(0xFFFFFFFF)), diffuse, diffuse2);
}


// apply diffuse lighting with respect to material's colouring
Colour8 applyDiffuse(const Vector8& lightRayStart, const Vector8& lightRayDir, const Light& currentLight, const Intersection8& intersects)
{
	__m256 diffuseReds = _mm256_i32gather_ps(scene.diffuseRed, intersects.materialIndex, 4);
	__m256 diffuseGreens = _mm256_i32gather_ps(scene.diffuseGreen, intersects.materialIndex, 4);
	__m256 diffuseBlues = _mm256_i32gather_ps(scene.diffuseBlue, intersects.materialIndex, 4);
	Colour8 diffuses = { diffuseReds, diffuseGreens, diffuseBlues };
	__m256 diffuse2Reds = _mm256_i32gather_ps(scene.diffuse2Red, intersects.materialIndex, 4);
	__m256 diffuse2Greens = _mm256_i32gather_ps(scene.diffuse2Green, intersects.materialIndex, 4);
	__m256 diffuse2Blues = _mm256_i32gather_ps(scene.diffuse2Blue, intersects.materialIndex, 4);
	Colour8 diffuse2s = { diffuse2Reds, diffuse2Greens, diffuse2Blues };
	__m256 offsetXs = _mm256_i32gather_ps(scene.offsetX, intersects.materialIndex, 4);
	__m256 offsetYs = _mm256_i32gather_ps(scene.offsetY, intersects.materialIndex, 4);
	__m256 offsetZs = _mm256_i32gather_ps(scene.offsetZ, intersects.materialIndex, 4);
	Vector8 offsets = { offsetXs, offsetYs, offsetZs };
	__m256 sizes = _mm256_i32gather_ps(scene.size, intersects.materialIndex, 4);

	__m256i types = _mm256_i32gather_epi32((int*) scene.materialType, intersects.materialIndex, 4);

	Colour8 intensitys = { _mm256_set1_ps(currentLight.intensity.red), _mm256_set1_ps(currentLight.intensity.green), _mm256_set1_ps(currentLight.intensity.blue) };

	// angle between direction to light and normal
	__m256 angleBetweenLightAndNormals = dot(lightRayDir, intersects.normal);

	// find colour at the intersection point
	Colour8 diffuse = diffuses;
	Colour8 checker = applyCheckerboard(intersects, diffuses, diffuse2s, offsets, sizes);
	Colour8 circles = applyCircles(intersects, diffuses, diffuse2s, offsets, sizes);
	Colour8 wood = applyWood(intersects, diffuses, diffuse2s, offsets, sizes);

	__m256i isDiffuse = types == (_mm256_set1_epi32((int)TextureType::GOURAUD));
	__m256i isChecker = types == (_mm256_set1_epi32((int)TextureType::CHECKERBOARD));
	__m256i isCircles = types == (_mm256_set1_epi32((int)TextureType::CIRCLES));
	__m256i isWood = types == (_mm256_set1_epi32((int)TextureType::WOOD));

	Colour8 outputs = select(isDiffuse, diffuse, outputs);
	outputs = select(isChecker, checker, outputs);
	outputs = select(isCircles, circles, outputs);
	outputs = select(isWood, wood, outputs);

	// ensure diffuse (lambert) value isn't less than zero
	__m256 lamberts = _mm256_max_ps(angleBetweenLightAndNormals, _mm256_setzero_ps());

	// final ambient value is multiplied by the intensity/colour of the light and the material's colour at this point
	outputs = lamberts * intensitys * outputs;

	return outputs;
}


// Blinn 
// The direction of Blinn is exactly at mid point of the light ray and the view ray. 
// We compute the Blinn vector and then we normalize it then we compute the coeficient of blinn
// which is the specular contribution of the current light.
Colour8 applySpecular(const Vector8& lightRayStart, const Vector8& lightRayDir, const Light& currentLight, const __m256 fLightProjections, 
	const Vector8& viewRayStart, const Vector8& viewRayDir, const Intersection8& intersect)
{
	// calculation of specular highlight (using Blinn)
	Vector8 blinnDirs = lightRayDir - viewRayDir;
	__m256 blinns = invsqrtf(blinnDirs.dot()) * _mm256_max_ps(fLightProjections - intersect.viewProjection, _mm256_setzero_ps());
	__m256 powers = _mm256_i32gather_ps(scene.power, intersect.materialIndex, 4);
	blinns = _mm256_pow_ps(blinns, powers);

	__m256 specularReds = _mm256_i32gather_ps(scene.specularRed, intersect.materialIndex, 4);
	__m256 specularGreens = _mm256_i32gather_ps(scene.specularGreen, intersect.materialIndex, 4);
	__m256 specularBlues = _mm256_i32gather_ps(scene.specularBlue, intersect.materialIndex, 4);

	Colour8 speculars = { specularReds, specularGreens, specularBlues };
	Colour8 intensitys = { _mm256_set1_ps(currentLight.intensity.red), _mm256_set1_ps(currentLight.intensity.green), _mm256_set1_ps(currentLight.intensity.blue) };

	// final specular value is multiplied by the amount of specular reflection of the material and the intensity/colour of the light
	return blinns * speculars * intensitys;
}


__m256 applySoftShadow(const Vector8& lightRayStart, const Vector8& lightRayDir, const Light& currentLight, const Intersection8& intersects)
{
	const Vector8 lightPoss(currentLight.pos.x, currentLight.pos.y, currentLight.pos.z);
	const __m256 ones = _mm256_set1_ps(1.0f);
	const __m256 tens = _mm256_set1_ps(10.0f);
	const __m256 EPSILONs = _mm256_set1_ps(EPSILON);

	// softness of shadows (smaller is softer)
	const __m256 SOFTNESS = _mm256_set1_ps(4.0f);

	// distance between light and point of intersection
	const __m256 distanceBetweenLightAndIntersection = (intersects.pos - lightPoss).length();

	// return value = light reaching intersection point (defaults to fully lit)
	__m256 res = ones;

	// distance along ray from light
	__m256 t = _mm256_setzero_ps();

	__m256 done2 = _mm256_setzero_ps();

	bool done[8] = { false };

	// loop until we've travelled close enough to intersection
	for (int steps = 0; steps < MAX_MARCH_STEPS; steps++)
	{
		// find distance to closest object
		__m256 dists = distance(lightPoss - lightRayDir * t);

		__m256 tSmallerThanDistance = t < distanceBetweenLightAndIntersection - tens;

		done2 = done2 | !tSmallerThanDistance;

		__m256 distLessThanEpsilon = dists < EPSILONs;

		res = select(!done2 & distLessThanEpsilon, _mm256_setzero_ps(), res);

		done2 = done2 | distLessThanEpsilon;

		res = select(done2, res, _mm256_min_ps(res, SOFTNESS * dists / (distanceBetweenLightAndIntersection - t)));

		t = t + dists;

		if (_mm256_test_all_ones(_mm256_castps_si256(done2))) break;
	}

	// return amount of light reaching point (1.0 = all light, 0.0 = none)
	return res;
}


// apply ambient occlusion
__m256 applyAmbientOcclusion(const Intersection8& intersect)
{
	// amount of ambient occlusion (none to begin with)
	__m256 occlusion = _mm256_setzero_ps();

	const __m256 ones = _mm256_set1_ps(1.0f);
	const __m256 twos = _mm256_set1_ps(2.0f);

	// strength of occlusion (smaller is stronger, doubles each successive step from the normal)
	__m256 k = ones;

	// distance along normal from intersection point
	__m256 t = _mm256_setzero_ps();

	const __m256 STEP_SIZEs = _mm256_set1_ps(AMBIENT_OCCLUSION_STEP_SIZE);
	const __m256 STRENGTHs = _mm256_set1_ps(AMBIENT_OCCLUSION_STRENGTH);

	// calculate occlision a fixed number of times at increasing distance along the normal
	for (int i = 1; i < AMBIENT_OCCLUSION_STEPS; i++)
	{
		// increase distance along the normal
		t = t + STEP_SIZEs;

		// calculate occlusion
		occlusion = occlusion + (ones / k) * (t - distance(intersect.pos + intersect.normal * t));

		// decrease strength of occlusion at next sample point
		k = k * twos;
	}

	// clamp occlusion amount to be between 0.0 and 1.0
	occlusion = _mm256_min_ps(_mm256_max_ps(occlusion, _mm256_setzero_ps()), ones);

	// return the amount of light getting through
	return ones - STRENGTHs * occlusion;
}


// apply diffuse and specular lighting contributions for all lights in scene taking shadowing into account
Colour8 applyLighting(const Scene& scene, const Vector8& viewRayPos, const Vector8& viewRayDir, const Intersection8& intersects)
{
	// colour to return (starts as black)
	Colour8 outputs = { 0 };

	// same starting point for each light ray
	Vector8 lightRayStart = intersects.pos;

	// loop through all the lights
	for (unsigned int j = 0; j < scene.numLights; ++j)
	{
		// get reference to current light
		const Light& currentLight = scene.lightContainer[j];

		Vector8 currentLightPos(currentLight.pos.x, currentLight.pos.y, currentLight.pos.z);

		// light ray direction need to equal the normalised vector in the direction of the current light
		// as we need to reuse all the intermediate components for other calculations, 
		// we calculate the normalised vector by hand instead of using the normalise function
		Vector8 lightRayDir = currentLightPos - intersects.pos;
		__m256 angleBetweenLightAndNormals = dot(lightRayDir, intersects.normal);

		// skip this light if it's behind the object (ie. both light and normal pointing in the same direction)
		__m256 lessThanZero = _mm256_cmp_ps(angleBetweenLightAndNormals, _mm256_setzero_ps(), _CMP_LT_OQ);

		// distance to light from intersection point (and it's inverse)
		__m256 lightDists = _mm256_sqrt_ps(lightRayDir.dot());
		__m256 invLightDists = _mm256_set1_ps(1.0f) / lightDists;

		// light ray projection
		__m256 lightProjections = invLightDists * angleBetweenLightAndNormals;

		// normalise the light direction
		lightRayDir = lightRayDir * invLightDists;

		// calculate all the lighting components
		Colour8 contributionDiffuse = applyDiffuse(lightRayStart, lightRayDir, currentLight, intersects);
		Colour8 contributionSpecular = applySpecular(lightRayStart, lightRayDir, currentLight, lightProjections, viewRayPos, viewRayDir, intersects);
		__m256 contributionSoftShadow = applySoftShadow(lightRayStart, lightRayDir, currentLight, intersects);
		__m256 contributionAO = applyAmbientOcclusion(intersects);

		// add them all together
		Colour8 contributions = (contributionDiffuse + contributionSpecular) * contributionSoftShadow * contributionAO;

		outputs = select(lessThanZero, outputs, outputs + contributions);
	}

	return outputs;
}