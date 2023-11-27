/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __SCENE_OBJECTS_H
#define __SCENE_OBJECTS_H

#include "Colour.h"
#include "Primitives.h"

#pragma warning(disable: 26495)

// type of colouring/texturing
enum class TextureType { GOURAUD, CHECKERBOARD, CIRCLES, WOOD };

// material
typedef struct Material
{
	TextureType type;			// type of colouring/texturing 

	Colour diffuse;				// diffuse colour
	Colour diffuse2;			// second diffuse colour, only for checkerboard types

	Vector offset;				// offset of generated texture
	float size;					// size of generated texture

	Colour specular;			// colour of specular lighting
	float power;				// power of specular reflection

	float reflection;			// reflection amount
} Material;


// light object
typedef struct Light
{
	Point pos;					// location
	Colour intensity;			// brightness and colour
} Light;


// sphere object
typedef struct Sphere
{
	Point pos;					// a point on the plane
	float size;					// radius of sphere
} Sphere;

// plane object
typedef struct Plane
{
	Point pos;					// a point on the plane
	Vector normal;				// normal of the plane
} Plane;

// (axis-aligned) box object
typedef struct Box
{
	Point pos;
	Vector halfSize;
} Box;

// type of scene object
enum class PrimitiveType { NONE, SPHERE, PLANE, BOX };

// way to combine the distance field for this object
enum class Operator { UNION, SUBTRACTION, INTERSECTION, SMOOTH };

// combined datastructure for all visible scene objects
typedef struct SceneObject
{
	SceneObject() : type(PrimitiveType::NONE), materialId(-1), op(Operator::UNION) { }

	PrimitiveType type;
	union
	{
		Sphere sphere;
		Plane plane;
		Box box;
	};
	unsigned int materialId;
	Operator op;
} SceneObject;

#endif // __SCENE_OBJECTS_H
