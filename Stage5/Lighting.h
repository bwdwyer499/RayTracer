/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __LIGHTING_H
#define __LIGHTING_H

#include "Scene.h"
#include "Intersection.h"
#include "PrimitivesSIMD.h"

// apply diffuse and specular lighting contributions for all lights in scene taking shadowing into account
Colour8 applyLighting(const Scene& scene, const Vector8& viewRayStart, const Vector8& viewRayDir, const Intersection8& intersect);


#endif // __LIGHTING_H
