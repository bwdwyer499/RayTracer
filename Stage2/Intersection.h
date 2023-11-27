/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __INTERSECTION_H
#define __INTERSECTION_H

#include "Scene.h"
#include "SceneObjects.h"

// all pertinant information about an intersection of a ray with an object
typedef struct Intersection
{
	Point pos;											// point of intersection
	Vector normal;										// normal at point of intersection
	float viewProjection;								// view projection 
	Material* material;									// material of object
} Intersection;

// calculate collision normal, viewProjection, object's material, and test to see if inside collision object
void calculateIntersectionResponse(const Scene& scene, const Ray& viewRay, int index, Intersection& intersect);

// find intersection with object along a ray
Point marchRay(Ray viewRay, int& index);

// trace ray and calculate colour information (including reflection)
Colour traceRay(const Scene& scene, Ray viewRay);

#endif // __INTERSECTION_H
