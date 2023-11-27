/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __LIGHTING_H
#define __LIGHTING_H

#include "Scene.h"
#include "Intersection.h"

// apply diffuse and specular lighting contributions for all lights in scene taking shadowing into account
Colour applyLighting(const Scene& scene, const Ray& viewRay, const Intersection& intersect); 


#endif // __LIGHTING_H
