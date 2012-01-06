#pragma once

#include <device_functions.h>
#include <vector_functions.h>
#include <sm_11_atomic_functions.h>		// atomicAdd()

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define DEVICE extern "C" __device__ __forceinline__
#define DEVICE2 __device__ __forceinline__

typedef unsigned int uint;
typedef unsigned short uint16;

typedef __int64 int64;
typedef unsigned __int64 uint64;

// Patches missing __hiloint2longlong function in CUDA library.
typedef volatile union {
	int64 ll;
	uint64 ull;
	double d;
	uint u[2];
} ConvertType64;

// Retrieve numBits bits from x starting at bit.
DEVICE2 uint bfe(uint x, uint bit, uint numBits) {
	uint ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;
}

// Provide a simple 64bit version.
DEVICE2 uint bfe(uint2 x, uint bit, uint numBits) {
	ConvertType64 t;
	t.u[0] = x.x;
	t.u[1] = x.y;
	uint shifted = (uint)(t.ull>> bit);
	return shifted & ((1<< numBits) - 1);
}

// Insert the first numBits of y into x starting at bit.
DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}

// Same syntax as __byte_perm, but without nvcc's __byte_perm bug that masks all
// non-immediate index arguments by 0x7777.
DEVICE uint prmt(uint a, uint b, uint index) {
	uint ret;
	asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
	return ret;
}

// Count trailing zeros - returns the number of consecutive zeroes from the 
// least significant bit.
DEVICE uint ctz(uint x) {
	return __clz(__brev(x));
}


// Put a float into radix order.
DEVICE uint FloatToUint(float f) {
	int x = __float_as_int(f);
	int bits = x>> 31;

	int flipped = x ^ (0x7fffffff & bits);

	int adjusted = 0x80000000 ^ flipped;

	uint u = (uint)adjusted;
	return u;
}

// Put a radix-order uint back into a float.
DEVICE float UintToFloat(uint u) {
	int adjusted = (int)u;

	// Negative now has high bit set, positive has high bit clear.
	int flipped = adjusted - 0x80000000;

	// Fill the register with set bits if negative.	
	int bits = flipped>> 31;

	int x = flipped ^ (0x7fffffff & bits);

	float f = __int_as_float(x);
	return f;
}

// Return a pair of integers for the radix-order double.
DEVICE uint2 DoubleToUint2(double d) {
	// Get the high and low parts of the double
	int high = __double2hiint(d);
	int low = __double2loint(d);

	// Right shift the high part to fill a register with 1s (if negative) or 0s
	// (if non-negative).
	int bits = high>> 31;
	
	// Flip all but the sign bit by bits.
	low ^= bits;
	high ^= (0x7fffffff & bits);

	// Flip the most significant bit to put negatives in the bottom half of the
	// range and positives in the top half of the range.
	high ^= 0x80000000;

	return make_uint2((uint)low, (uint)high);
}

DEVICE2 uint ConvertToRadix(uint x) {
	return x; 
}
DEVICE2 uint ConvertToRadix(int x) { 
	return 0x80000000 ^ (uint)x; 
}
DEVICE2 uint ConvertToRadix(float x) { 
	return FloatToUint(x);
}
DEVICE2 uint2 ConvertToRadix(uint64 x) {
	ConvertType64 t;
	t.ull = x;
	return make_uint2(t.u[0], t.u[1]);	
}
DEVICE2 uint2 ConvertToRadix(int64 x) {
	ConvertType64 t;
	t.ll = x;
	return make_uint2(t.u[0], 0x80000000 ^ t.u[1]);
}
DEVICE2 uint2 ConvertToRadix(double x) {
	return DoubleToUint2(x);
}

template<typename T>
DEVICE2 inline T DivUp(T num, T den) {
	return (num + den - 1) / den;
}
template<typename T>
DEVICE2 T RoundUp(T x, T y) {
	return ~(y - 1) & (x + y - 1);
}
template<typename T>
DEVICE2 T RoundDown(T x, T y) {
	return ~(y - 1) & x;
}
