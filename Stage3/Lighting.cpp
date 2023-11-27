/*  The following code is a VERY heavily modified from code originally sourced from:
Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#include "Lighting.h"
#include "Colour.h"
#include "Intersection.h"
#include "Texturing.h"
#include "Distance.h"


// apply diffuse lighting with respect to material's colouring
Colour applyDiffuse(const Ray& lightRay, const Light& currentLight, const Intersection& intersect)
{
	Colour output;

	// find colour at the intersection point
	switch (intersect.material->type)
	{
	case TextureType::GOURAUD:
		output = intersect.material->diffuse;
		break;
	case TextureType::CHECKERBOARD:
		output = applyCheckerboard(&intersect);
		break;
	case TextureType::CIRCLES:
		output = applyCircles(&intersect);
		break;
	case TextureType::WOOD:
		output = applyWood(&intersect);
		break;
	}

	// angle between direction to light and normal
	float angleBetweenLightAndNormal = lightRay.dir * intersect.normal;

	// ensure diffuse (lambert) value isn't less than zero
	float lambert = fmaxf(angleBetweenLightAndNormal, 0.0f);

	// final ambient value is multiplied by the intensity/colour of the light and the material's colour at this point
	return lambert * currentLight.intensity * output;
}


// Blinn 
// The direction of Blinn is exactly at mid point of the light ray and the view ray. 
// We compute the Blinn vector and then we normalize it then we compute the coeficient of blinn
// which is the specular contribution of the current light.
Colour applySpecular(const Ray& lightRay, const Light& currentLight, const float fLightProjection, const Ray& viewRay, const Intersection& intersect)
{
	// calculation of specular highlight (using Blinn)
	Vector blinnDir = lightRay.dir - viewRay.dir;
	float blinn = invsqrtf(blinnDir.dot()) * fmaxf(fLightProjection - intersect.viewProjection, 0.0f);
	blinn = powf(blinn, intersect.material->power);

	// final specular value is multiplied by the amount of specular reflection of the material and the intensity/colour of the light
	return blinn * intersect.material->specular * currentLight.intensity;
}


// apply soft shadowing
float applySoftShadow(const Ray& lightRay, const Light& currentLight, const Intersection& intersect)
{
	// softness of shadows (smaller is softer)
	const float SOFTNESS = 4.0f;

	// distance between light and point of intersection
	const float distanceBetweenLightAndIntersection = (intersect.pos - currentLight.pos).length();

	// return value = light reaching intersection point (defaults to fully lit)
	float res = 1.0f;

	// distance along ray from light
	float t = 0.0f;

	int steps = 0;

	// loop until we've travelled close enough to intersection
	while (t < distanceBetweenLightAndIntersection - 10.0f && steps < MAX_MARCH_STEPS)
	{
		// find distance to closest object
		float dist = distance(currentLight.pos - lightRay.dir * t);

		// if collision with object detected, intersection point is in full shadow
		if (dist < EPSILON) return 0.0f;

		// remember closest point
		res = fminf(res, SOFTNESS * dist / (distanceBetweenLightAndIntersection - t));

		// move along ray
		t += dist;

		steps++;
	}

	// return amount of light reaching point (1.0 = all light, 0.0 = none)
	return res;
}


// apply ambient occlusion
float applyAmbientOcclusion(const Intersection& intersect)
{
	// amount of ambient occlusion (none to begin with)
	float occlusion = 0.0f;

	// strength of occlusion (smaller is stronger, doubles each successive step from the normal)
	float k = 1.0f;

	// distance along normal from intersection point
	float t = 0.0f;

	// calculate occlision a fixed number of times at increasing distance along the normal
	for (int i = 1; i < AMBIENT_OCCLUSION_STEPS; i++)
	{
		// increase distance along the normal
		t += AMBIENT_OCCLUSION_STEP_SIZE;

		// calculate occlusion
		occlusion += (1.0f / k) * (t - distance(intersect.pos + intersect.normal * t));

		// decrease strength of occlusion at next sample point
		k *= 2.0f;
	}

	// clamp occlusion amount to be between 0.0 and 1.0
	occlusion = fminf(1.0f, fmaxf(0.0f, occlusion));

	// return the amount of light getting through
	return 1.0f - AMBIENT_OCCLUSION_STRENGTH * occlusion;
}


// apply diffuse and specular lighting contributions for all lights in scene taking shadowing into account
Colour applyLighting(const Scene& scene, const Ray& viewRay, const Intersection& intersect)
{
	// colour to return (starts as black)
	Colour output(0.0f, 0.0f, 0.0f);

	// same starting point for each light ray
	Ray lightRay = { intersect.pos };

	// loop through all the lights
	for (unsigned int j = 0; j < scene.numLights; ++j)
	{
		// get reference to current light
		const Light& currentLight = scene.lightContainer[j];

		// light ray direction need to equal the normalised vector in the direction of the current light
		// as we need to reuse all the intermediate components for other calculations, 
		// we calculate the normalised vector by hand instead of using the normalise function
		lightRay.dir = currentLight.pos - intersect.pos;
		float angleBetweenLightAndNormal = lightRay.dir * intersect.normal;

		// skip this light if it's behind the object (ie. both light and normal pointing in the same direction)
		if (angleBetweenLightAndNormal <= 0.0f)
		{
			continue;
		}

		// distance to light from intersection point (and it's inverse)
		float lightDist = sqrtf(lightRay.dir.dot());
		float invLightDist = 1.0f / lightDist;

		// light ray projection
		float lightProjection = invLightDist * angleBetweenLightAndNormal;

		// normalise the light direction
		lightRay.dir = lightRay.dir * invLightDist;

		// add diffuse lighting from colour / texture
		Colour contribution = applyDiffuse(lightRay, currentLight, intersect);

		// add specular lighting
		contribution += applySpecular(lightRay, currentLight, lightProjection, viewRay, intersect);

		// multiply (soft) shadow amount
		contribution *= applySoftShadow(lightRay, currentLight, intersect);

		// multiply ambient occlusion amount
		contribution *= applyAmbientOcclusion(intersect);

		output += contribution;
	}

	return output;
}
