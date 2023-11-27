// SIMD versions of (some of the) helper functions from primitives.h

#ifndef __PRIMITIVES_SIMD_H
#define __PRIMITIVES_SIMD_H

#include <immintrin.h>

// a bunch of operators / functions to replace nasty instrinsics
__forceinline __m256 operator - (const __m256 x, const __m256 y) { return _mm256_sub_ps(x, y); }
__forceinline __m256 operator + (const __m256 x, const __m256 y) { return _mm256_add_ps(x, y); }
__forceinline __m256 operator * (const __m256 x, const __m256 y) { return _mm256_mul_ps(x, y); }
__forceinline __m256 operator / (const __m256 x, const __m256 y) { return _mm256_div_ps(x, y); }
__forceinline __m256 operator < (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LT_OQ); }
__forceinline __m256 operator > (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GT_OQ); }
__forceinline __m256 operator <= (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LE_OQ); }
__forceinline __m256 operator >= (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GE_OQ); }
__forceinline __m256 operator & (const __m256 x, const __m256 y) { return _mm256_and_ps(x, y); }
__forceinline __m256 operator | (const __m256 x, const __m256 y) { return _mm256_or_ps(x, y); }
__forceinline __m256 operator ! (const __m256 x) { return _mm256_andnot_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF))); }
__forceinline __m256 operator == (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_EQ_OQ); }
__forceinline __m256 operator != (const __m256 x, const __m256 y) { return _mm256_cmp_ps(x, y, _CMP_NEQ_OQ); }

__forceinline __m256i operator & (const __m256i x, const __m256i y) { return _mm256_and_si256(x, y); }
__forceinline __m256i operator | (const __m256i x, const __m256i y) { return _mm256_or_si256(x, y); }
__forceinline __m256i operator ! (const __m256i x) { return _mm256_andnot_si256(x, _mm256_set1_epi32(0xFFFFFFFF)); }

__forceinline __m256i operator + (const __m256i x, const __m256i y) { return _mm256_add_epi32(x, y); }

__forceinline __m256 select(__m256 cond, __m256 ifTrue, __m256 ifFalse)
{
	return _mm256_blendv_ps(ifFalse, ifTrue, cond);
	//return _mm256_or_ps(_mm256_and_ps(cond, ifTrue), _mm256_andnot_ps(cond, ifFalse));
}

__forceinline __m256i select(__m256i cond, __m256i ifTrue, __m256i ifFalse)
{
	return _mm256_blendv_epi8(ifFalse, ifTrue, cond);
	//return _mm256_or_si256(_mm256_and_si256(cond, ifTrue), _mm256_andnot_si256(cond, ifFalse));
}


// absolute value
__forceinline __m256 abs(__m256 x)
{
	// -0.0f is 0x80000000 (i.e. the sign bit), so mask that bit away and you have an absolute value
	return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

	// this approach (and some others) would also be acceptable
	//const __m256 zeros = _mm256_set1_ps(0.0f);
	//__m256 lessThanZero = x < zeros;
	//return select(lessThanZero, zeros - x, x);
}

// sign (i.e. -1 when negative, 0 when 0, 1 when positive)
__forceinline __m256 sign(__m256 x)
{
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 ones = _mm256_set1_ps(1.0f);
	const __m256 minusOnes = _mm256_set1_ps(-1.0f);
	__m256 lessThanZero = x < zeros;
	__m256 greaterThanZero = x > zeros;

	return select(lessThanZero, minusOnes, select(greaterThanZero, ones, zeros));
}

__forceinline __m256 my_exp(__m256 x)
{
	// more accurate, but not SIMD -- other people had far more fun coming up with solutions to this than me
	for (int i = 0; i < 8; i++)
	{
		x.m256_f32[i] = expf(x.m256_f32[i]);
	}
	return x;

	//return _mm256_exp_ps(x);
}

// Represent 8 vectors in one struct
struct Vector8
{
	__m256 xs, ys, zs;

	__forceinline Vector8(float x, float y, float z)
	{
		xs = _mm256_set1_ps(x);
		ys = _mm256_set1_ps(y);
		zs = _mm256_set1_ps(z);
	}

	__forceinline Vector8(__m256 xsIn, __m256 ysIn, __m256 zsIn)
	{
		xs = xsIn;
		ys = ysIn;
		zs = zsIn;
	}

	__forceinline Vector8()
	{
		xs = ys = zs = _mm256_setzero_ps();
	}

	__forceinline __m256 length()
	{
		return _mm256_sqrt_ps(xs * xs + ys * ys + zs * zs);
	}

	__forceinline __m256 dot() const
	{
		return xs * xs + ys * ys + zs * zs;
	}
};

struct Colour8
{
	__m256 reds, greens, blues;

	inline __m256i convertToPixel(float exposure)
	{
		const __m256 scale = _mm256_set1_ps(255.0f);
		const __m256 ones = _mm256_set1_ps(1.0f);
		const __m256 exposures = _mm256_set1_ps(exposure);

		// one pixel fix here is reversing the order of the arguments in min
		__m256i b = _mm256_slli_epi32(_mm256_cvtps_epi32(_mm256_floor_ps(scale * _mm256_min_ps(ones, ones - my_exp(blues * exposures)))), 16);
		__m256i g = _mm256_slli_epi32(_mm256_cvtps_epi32(_mm256_floor_ps(scale * _mm256_min_ps(ones, ones - my_exp(greens * exposures)))), 8);
		__m256i r = _mm256_cvtps_epi32(_mm256_floor_ps(scale * _mm256_min_ps(ones, ones - my_exp(reds * exposures))));

		return b + g + r;
	}


	// convert colour to pixel (in 0x00BBGGRR format)
	inline __m256i convertToPixel()
	{
		const __m256 scale = _mm256_set1_ps(255.0f);
		const __m256 ones = _mm256_set1_ps(1.0f);

		__m256i b = _mm256_slli_epi32(_mm256_cvtps_epi32(scale * _mm256_min_ps(blues, ones)), 16);
		__m256i g = _mm256_slli_epi32(_mm256_cvtps_epi32(scale * _mm256_min_ps(greens, ones)), 8);
		__m256i r = _mm256_cvtps_epi32(scale * _mm256_min_ps(reds, ones));

		return b + g + r;
	}
};



// helper operators / functions for Vector8s
__forceinline Vector8 operator - (const Vector8& v1, const Vector8& v2) { return { v1.xs - v2.xs, v1.ys - v2.ys, v1.zs - v2.zs }; }
__forceinline Vector8 operator + (const Vector8& v1, const Vector8& v2) { return { v1.xs + v2.xs, v1.ys + v2.ys, v1.zs + v2.zs }; }
__forceinline Vector8 operator * (const Vector8& v1, const Vector8& v2) { return { v1.xs * v2.xs, v1.ys * v2.ys, v1.zs * v2.zs }; }
__forceinline Vector8 operator / (const Vector8& v1, const Vector8& v2) { return { v1.xs / v2.xs, v1.ys / v2.ys, v1.zs / v2.zs }; }

__forceinline Vector8 operator + (const Vector8& v, const __m256 x) { return { v.xs + x, v.ys + x, v.zs + x }; }
__forceinline Vector8 operator - (const Vector8& v, const __m256 x) { return { v.xs - x, v.ys - x, v.zs - x }; }
__forceinline Vector8 operator * (const Vector8& v, const __m256 x) { return { v.xs * x, v.ys * x, v.zs * x }; }
__forceinline Vector8 operator / (const Vector8& v, const __m256 x) { return { v.xs / x, v.ys / x, v.zs / x }; }

__forceinline Vector8 cross(const Vector8& v1, const Vector8& v2)
{
	return { v1.ys * v2.zs - v1.zs * v2.ys, v1.zs * v2.xs - v1.xs * v2.zs, v1.xs * v2.ys - v1.ys * v2.xs };
}

__forceinline __m256 dot(const Vector8& v1, const Vector8& v2)
{
	return v1.xs * v2.xs + v1.ys * v2.ys + v1.zs * v2.zs;
}

__forceinline Vector8 select(__m256 cond, const Vector8& v1, const Vector8& v2)
{
	return Vector8(select(cond, v1.xs, v2.xs), select(cond, v1.ys, v2.ys), select(cond, v1.zs, v2.zs));
}

__forceinline Vector8 select(__m256i cond, const Vector8& v1, const Vector8& v2)
{
	return Vector8(select(_mm256_castsi256_ps(cond), v1.xs, v2.xs), select(_mm256_castsi256_ps(cond), v1.ys, v2.ys), select(_mm256_castsi256_ps(cond), v1.zs, v2.zs));
}

__forceinline Vector8 max(const Vector8& v1, const Vector8& v2)
{
	return Vector8(_mm256_max_ps(v1.xs, v2.xs), _mm256_max_ps(v1.ys, v2.ys), _mm256_max_ps(v1.zs, v2.zs));
}

__forceinline Vector8 min(const Vector8& v1, const Vector8& v2)
{
	return Vector8(_mm256_min_ps(v1.xs, v2.xs), _mm256_min_ps(v1.ys, v2.ys), _mm256_min_ps(v1.zs, v2.zs));
}

__forceinline Vector8 abs(const Vector8& v)
{
	return Vector8(abs(v.xs), abs(v.ys), abs(v.zs));
}

__forceinline __m256 hmax(const Vector8& v)
{
	return _mm256_max_ps(v.xs, _mm256_max_ps(v.ys, v.zs));
}

inline Vector8 normalise(const Vector8& x)
{

	// for perfect imagemagick diffs, need to match accuracy of original code
	const __m256 ones = _mm256_set1_ps(1.0f);
	return x * (ones / _mm256_sqrt_ps(x.dot()));

	// rsqrt would actually be fine visually, but does cause minor differences in accuracy
	//return x * _mm256_rsqrt_ps(x.dot());
}

inline __m256 clamp01(__m256 x)
{
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 ones = _mm256_set1_ps(1.0f);
	return _mm256_min_ps(ones, _mm256_max_ps(zeros, x));
}

// linearly interpolate between two values
inline __m256 lerp(__m256 x, __m256 y, __m256 t)
{
	const __m256 ones = _mm256_set1_ps(1.0f);
	return x * (ones - t) + y * t;
}

// structure combining a distance and an index and a couple of helper functions
struct DistanceAndIndex8
{
	__m256 dist;
	__m256i index;
};

#endif

