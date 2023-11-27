#include "Texturing.h"
#include "Colour.h"
#include "Intersection.h"

// apply computed checkerboard texture
Colour applyCheckerboard(const Intersection* intersect)
{
	Point p = (intersect->pos - intersect->material->offset) / intersect->material->size;

	int which = (int(floorf(p.x)) + int(floorf(p.y)) + int(floorf(p.z))) & 1;

	return (which ? intersect->material->diffuse : intersect->material->diffuse2);
}


// apply computed circular texture
Colour applyCircles(const Intersection* intersect)
{
	Point p = (intersect->pos - intersect->material->offset) / intersect->material->size;

	int which = int(floorf(sqrtf(p.x*p.x + p.y*p.y + p.z*p.z))) & 1;

	return (which ? intersect->material->diffuse : intersect->material->diffuse2);
}


// apply computed wood grain texture
Colour applyWood(const Intersection* intersect)
{
	Point p = (intersect->pos - intersect->material->offset) / intersect->material->size;

	// squiggle up where the point is
	p = { p.x * cosf(p.y * 0.666f) * sinf(p.z * 1.027f), cosf(p.x) * p.y * sinf(p.z * 1.212f), cosf(p.x * 1.471f) * cosf(p.y * 0.793f) * p.z };

	int which = int(floorf(sqrtf(p.x*p.x + p.y*p.y + p.z*p.z))) & 1;

	return (which ? intersect->material->diffuse : intersect->material->diffuse2);
}
