/*  The following code is a VERY heavily modified from code originally sourced from:
Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#define TARGET_WINDOWS

#pragma warning(disable: 4996)
#pragma warning(disable: 26495)
#include <stdio.h>
#include "Timer.h"
#include "Primitives.h"
#include "Scene.h"
#include "Lighting.h"
#include "Intersection.h"
#include "ImageIO.h"
#include "Distance.h"


// global pixel buffer for rendered image
unsigned int buffer[MAX_WIDTH * MAX_HEIGHT];

// global scene file (used to simplify distance functions)
Scene scene;


// render scene at given width and height and anti-aliasing level
unsigned int renderSection(Scene& scene, const int width, const int height, const int aaLevel, const int blockSize,
	bool debugProgress, const bool testMode, const float testColour, const unsigned int colourMask,
	unsigned int* currentBlockShared, unsigned int* out)
{
	// angle between each successive ray cast (per pixel, anti-aliasing uses a fraction of this)
	const float dirStepSize = 1.0f / (0.5f * width / tanf(PIOVER180 * 0.5f * scene.cameraFieldOfView));

	// calculate exactly how many blocks are needed (and deal with cases where the blockSize doesn't exactly divide)
	unsigned int blocksWide = (width - 1) / blockSize + 1;
	unsigned int blocksHigh = (height - 1) / blockSize + 1;
	unsigned int blocksTotal = blocksWide * blocksHigh;

	// count of samples rendered
	unsigned int samplesRendered = 0;

	// current block index
	unsigned int currentBlock;

	while ((currentBlock = InterlockedIncrement(currentBlockShared)) < blocksTotal)
	{
		// block x,y position
		const int bx = currentBlock % blocksWide;
		const int by = currentBlock / blocksWide;

		// block coordinates (making sure not to exceed image bounds with non-divisible block sizes)
		const int xMin = bx * blockSize - width / 2;
		const int xMax = (std::min)(xMin + blockSize, width / 2);
		const int yMin = by * blockSize - height / 2;
		const int yMax = (std::min)(yMin + blockSize, height / 2);

		// debug
		if (debugProgress) printf("thread rendering: block %d [%d,%d] (%d,%d)->(%d,%d) => %p\n", currentBlock, bx, by, xMin, yMin, xMax, yMax, out);

		// calculate the array location of the start of the block
		unsigned int* outBlock = buffer + (long long int) width * by * blockSize + (long long int) bx * blockSize;

		// jump required to get to the start of the next line of the block
		unsigned int outJump = width - (xMax - xMin);

		// loop through all the pixels
		for (int y = yMin; y < yMax; ++y)
		{
			for (int x = xMin; x < xMax; ++x)
			{
				Colour output(0.0f, 0.0f, 0.0f);

				// calculate multiple samples for each pixel
				const float sampleStep = 1.0f / aaLevel, sampleRatio = 1.0f / (aaLevel * aaLevel);

				// loop through all sub-locations within the pixel
				for (float fragmentx = float(x); fragmentx < x + 1.0f; fragmentx += sampleStep)
				{
					for (float fragmenty = float(y); fragmenty < y + 1.0f; fragmenty += sampleStep)
					{
						// direction of default forward facing ray
						Vector dir = { fragmentx * dirStepSize, fragmenty * dirStepSize, 1.0f };

						// rotated direction of ray
						Vector rotatedDir = {
							dir.x * cosf(scene.cameraRotation) - dir.z * sinf(scene.cameraRotation),
							dir.y,
							dir.x * sinf(scene.cameraRotation) + dir.z * cosf(scene.cameraRotation) };

						// view ray starting from camera position and heading in rotated (normalised) direction
						Ray viewRay = { scene.cameraPosition, normalise(rotatedDir) };

						// follow ray and add proportional of the result to the final pixel colour
						output += sampleRatio * traceRay(scene, viewRay);

						// count this sample
						samplesRendered++;
					}
				}

				if (!testMode)
				{
					if (colourMask == 0)
					{
						// store saturated final colour value in image buffer
						*outBlock++ = output.convertToPixel(scene.exposure);
					}
					else
					{
						// with multiple threads this will tint the rendered output based on the thread number
						output.colourise(colourMask);
						*outBlock++ = output.convertToPixel(scene.exposure);
					}
				}
				else
				{
					// with multiple threads this should store a grey based on the thread number
					*outBlock++ = Colour(testColour, testColour, testColour).convertToPixel();
				}
			}

			outBlock += outJump;
		}
	}

	return samplesRendered;
}


struct ThreadParams
{
	Scene* scene;
	int width;
	int height;
	int aaLevel;
	int blockSize;
	bool debugProgress;
	bool testMode;
	float testColour;
	unsigned int colourMask;
	unsigned int* currentBlockShared;
	unsigned int* out;
};


// thread callback for rendering
DWORD __stdcall renderSectionThread(LPVOID inData)
{
	// cast the void* structure to something useful
	ThreadParams* params = (ThreadParams*)inData;

	// call the real render function
	int samplesRendered = renderSection(*params->scene, params->width, params->height, params->aaLevel, params->blockSize,
		params->debugProgress, params->testMode, params->testColour, params->colourMask, params->currentBlockShared, params->out);

	// exit with success, returning the number of samples rendered by this thread
	ExitThread(samplesRendered);
}


// render scene at given width and height and anti-aliasing level using a specified number of threads
unsigned int render(const unsigned int threadCount, Scene& scene, const int width, const int height, const int aaLevel, const int blockSize, const bool debugProgress, const bool testMode, const bool colourise)
{
	// count of samples rendered
	unsigned int samplesRendered = 0;

	// reserve space for threads and their parameters
	HANDLE* threads = new HANDLE[threadCount];
	ThreadParams* params = new ThreadParams[threadCount];

	// one less than the current block to render (shared between threads)
	unsigned int currentBlockShared = -1;

	// loop through all the threads
	for (unsigned int i = 0; i < threadCount; ++i)
	{
		// set up thread parameters
		params[i] = { &scene, width, height, aaLevel, blockSize, debugProgress, testMode, i / (float)threadCount, colourise ? (i % 8) : 0, &currentBlockShared, buffer };

		// start thread
		threads[i] = CreateThread(NULL, 0, renderSectionThread, (LPVOID)&params[i], 0, NULL);
	}

	// wait until all the threads are done
	for (unsigned int i = 0; i < threadCount; i++) 
	{
		if (threads[i] != NULL)
		{
			WaitForSingleObject(threads[i], INFINITE);

			// use the return value to calculate the total number of samples rendered
			DWORD retValue;
			if (GetExitCodeThread(threads[i], &retValue))
			{
				samplesRendered += retValue;
			}
		}
	}

	// clean up thread and param storage
	delete[] params;
	delete[] threads;

	// return cumulative samples rendered value
	return samplesRendered;
}


#pragma warning(disable: 6011)
// allocate space fro SoA, and copy values from AoS to SoA 
void simdifySceneContainer(Scene& scene)
{
	// helper size (so we don't just have 8 everywhere)
	unsigned int valuesPerVector = sizeof(__m256) / sizeof(float);

	// make SoA SIMD copies of scene objects

	// mathemagical way of calculating ceilf(scene.numSpheres / 8.0f)
	scene.numObjectsSIMD = (((int)scene.numObjects) - 1) / valuesPerVector + 1;

	// allocate the correct amount of space at the correct alignment for SIMD operations
	// setting of scene.sphereSize, .boxHalfSizeX, .boxHalfSizeY, .boxHalfSizeZ is implicit (due to the union) and not included
	scene.objectType = (__m256i*) _aligned_malloc(sizeof(__m256i) * scene.numObjectsSIMD, sizeof(__m256i));
	scene.objectPosX = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.objectPosY = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.objectPosZ = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.planeNormalX = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.planeNormalY = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.planeNormalZ = (__m256*) _aligned_malloc(sizeof(__m256) * scene.numObjectsSIMD, sizeof(__m256));
	scene.objectMaterialId = (__m256i*) _aligned_malloc(sizeof(__m256i) * scene.numObjectsSIMD, sizeof(__m256i));
	scene.objectOp = (__m256i*) _aligned_malloc(sizeof(__m256i) * scene.numObjectsSIMD, sizeof(__m256i));

	// initialise SoA structures
	for (unsigned int i = 0; i < scene.numObjectsSIMD * valuesPerVector; ++i)
	{
		// don't let the source index extend out of the AoS array
		// i.e. copy the last value into the extra array slots when numSpheres isn't exactly divisible by 8
		// pretty lazy way to fix this, but it works (and is the fastest approach in this case)
		int sourceIndex = i < scene.numObjects ? i : scene.numObjects - 1;

		scene.objectType[i / valuesPerVector].m256i_i32[i % valuesPerVector] = (unsigned int)scene.objectContainer[sourceIndex].type;
		scene.objectPosX[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.pos.x;
		scene.objectPosY[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.pos.y;
		scene.objectPosZ[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.pos.z;
		scene.planeNormalX[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.normal.x;
		scene.planeNormalY[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.normal.y;
		scene.planeNormalZ[i / valuesPerVector].m256_f32[i % valuesPerVector] = scene.objectContainer[sourceIndex].plane.normal.z;
		scene.objectMaterialId[i / valuesPerVector].m256i_i32[i % valuesPerVector] = scene.objectContainer[sourceIndex].materialId;
		scene.objectOp[i / valuesPerVector].m256i_i32[i % valuesPerVector] = (unsigned int)scene.objectContainer[sourceIndex].op;
	}
}


// read command line arguments, render, and write out BMP file
int main(int argc, char* argv[])
{
	int width = 1024;
	int height = 1024;
	int samples = 1;

	// rendering options
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	unsigned int threads = sysinfo.dwNumberOfProcessors;
	int times = 1;
	bool debugProgress = false;
	bool testMode = false;
	bool colourise = false;
	unsigned int blockSize = 16;

	// default input / output filenames
	const char* inputFilename = "Scenes/cornell.txt";

	char outputFilenameBuffer[1000];
	char* outputFilename = outputFilenameBuffer;

	// do stuff with command line args
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-size") == 0)
		{
			width = atoi(argv[++i]);
			height = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-samples") == 0)
		{
			samples = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-input") == 0)
		{
			inputFilename = argv[++i];
		}
		else if (strcmp(argv[i], "-output") == 0)
		{
			outputFilename = argv[++i];
		}
		else if (strcmp(argv[i], "-runs") == 0)
		{
			times = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-threads") == 0)
		{
			threads = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-colourise") == 0)
		{
			colourise = true;
		}
		else if (strcmp(argv[i], "-blockSize") == 0)
		{
			blockSize = atoi(argv[++i]);
		}
		else if (strcmp(argv[i], "-debugProgress") == 0)
		{
			debugProgress = true;
		}
		else if (strcmp(argv[i], "-testMode") == 0)
		{
			testMode = true;
		}
		else
		{
			fprintf(stderr, "unknown argument: %s\n", argv[i]);
		}
	}

	// nasty (and fragile) kludge to make an ok-ish default output filename (can be overriden with "-output" command line option)
	sprintf(outputFilenameBuffer, "Outputs/%s_%dx%dx%d_%s.bmp", (strrchr(inputFilename, '/') + 1), width, height, samples, (strrchr(argv[0], '\\') + 1));

	// read scene file
	if (!init(inputFilename, scene))
	{
		fprintf(stderr, "Failure when reading the Scene file.\n");
		return -1;
	}

	// make a copy of AoS structures as SoA
	simdifySceneContainer(scene);

	// total time taken to render all runs (used to calculate average)
	int totalTime = 0;
	int samplesRendered = 0;
	for (int i = 0; i < times; i++)
	{
		Timer timer;																										// create timer
		samplesRendered = render(threads, scene, width, height, samples, blockSize, debugProgress, testMode, colourise);	// raytrace scene
		timer.end();																										// record end time
		totalTime += timer.getMilliseconds();																				// record total time taken
	}

	// output timing information (times run and average)
	printf("rendered %d samples using %d threads, average time taken (%d run(s)): %.1fms\n", samplesRendered, threads, times, totalTime / (float) times);

	// output BMP file
	write_bmp(outputFilename, buffer, width, height, width);
}
