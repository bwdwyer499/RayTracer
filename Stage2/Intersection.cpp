/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#include "Intersection.h"
#include "DFPrimitives.h"
#include "Lighting.h"
#include "Distance.h"

Vector calculateNormal(const Point& p)
{
	Vector n(distance(Point(p.x + EPSILON, p.y, p.z)) - distance(Point(p.x - EPSILON, p.y, p.z)),
		distance(Point(p.x, p.y + EPSILON, p.z)) - distance(Point(p.x, p.y - EPSILON, p.z)),
		distance(Point(p.x, p.y, p.z + EPSILON)) - distance(Point(p.x, p.y, p.z - EPSILON)));

	return normalise(n);
}

// calculate collision normal, viewProjection, object's material, and test to see if inside collision object
void calculateIntersectionResponse(const Scene& scene, const Ray& viewRay, int index, Intersection& intersect)
{
	intersect.normal = calculateNormal(intersect.pos);
	intersect.material = &scene.materialContainer[scene.objectContainer[index].materialId];

	// calculate view projection
	intersect.viewProjection = viewRay.dir * intersect.normal; 
}


// find intersection with object along a ray
Point marchRay(Ray viewRay, int& index)
{
	Point currentPoint = viewRay.start;
	DistanceAndIndex dai = { FLT_MAX, -1 };

	// progress a little along the ray to make sure we aren't too close to the previous object
	currentPoint = currentPoint + viewRay.dir * EPSILON;

	// loop until too many steps taken or collided with object
	for (int steps = 0; steps < MAX_MARCH_STEPS; ++steps)
	{
		// find distance to nearest object (and it's index)
		dai = distanceAndIndex(currentPoint);

		// if we are very close to something, consider that an intersection and exit
		if (dai.dist < EPSILON)
		{
			currentPoint = currentPoint + viewRay.dir * (dai.dist - EPSILON);
			break;
		}

		// make sure than step size isn't too small
		if (dai.dist < MIN_MARCH_STEP) dai.dist = MIN_MARCH_STEP;

		// move the point
		currentPoint = currentPoint + viewRay.dir * dai.dist;
	}

	// return the index of the closest object
	index = dai.index;

	return currentPoint;
}


// trace ray and calculate colour information (including reflection)
Colour traceRay(const Scene& scene, Ray viewRay)
{
	Colour output(0.0f, 0.0f, 0.0f); 								// colour value to be output
	float coef = 1.0f;												// amount of ray left to transmit
	Intersection intersect = {  };									// properties of current intersection

	bool insideObject = false;

	// loop until reached maximum ray cast limit (unless loop is broken out of)
	for (int level = 0; level < MAX_RAYS_CAST; ++level)
	{
		// check for intersections between the view ray and any of the objects in the scene
		int index = -1;
		intersect.pos = marchRay(viewRay, index);

		// exit the loop if no intersection found
		if (index < 0) break;

		// calculate response to collision: ie. get normal at point of collision and material of object
		calculateIntersectionResponse(scene, viewRay, index, intersect);

		// apply colour
		output += coef * applyLighting(scene, viewRay, intersect);

		// if object has reflection component, adjust the view ray and coefficent of calculation and continue looping
		if (intersect.material->reflection)
		{
			viewRay = { intersect.pos, viewRay.dir - (intersect.normal * intersect.viewProjection * 2.0f) };
			coef *= intersect.material->reflection;
		}
		else
		{
			// if no reflection then finish looping (cast no more rays)

			return output;
		}
	}

	// if the calculation coefficient is non-zero, read from the environment map
	if (coef > 0.0f)
	{
		Material& currentMaterial = scene.materialContainer[scene.skyboxMaterialId];

		output += coef * currentMaterial.diffuse;
	}

	return output;
}