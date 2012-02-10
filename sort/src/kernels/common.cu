#pragma once

#define MAX(a, b) (((a) >= (b)) ? (a) : (b))
#define MIN(a, b) (((a) <= (b)) ? (a) : (b))

#define ROUND_UP(a, b) (~((b) - 1) & ((a) + (b) - 1))
#define ROUND_DOWN(a, b) (~(b - 1) & a)

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5


#ifndef NUM_WARPS
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#else
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)
#endif

#ifndef NO_VIDEO_INSTRUCTIONS
#define USE_VIDEO_INSTRUCTIONS
#endif

#include <device_functions.h>
#include <vector_functions.h>
#include <sm_11_atomic_functions.h>		// atomicAdd()

#define DEVICE extern "C" __device__ __forceinline__
#define DEVICE2 __device__ __forceinline__

typedef unsigned int uint;
typedef unsigned short uint16;
typedef __int64 int64;
typedef unsigned __int64 uint64;


// retrieve numBits bits from x starting at bit
DEVICE uint bfe(uint x, uint bit, uint numBits) {
	uint ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


// insert the first numBits of y into x starting at bit
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

DEVICE2 uint shl_add(uint a, uint b, uint c) {
#ifdef USE_VIDEO_INSTRUCTIONS
	uint ret;
	asm("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
		"=r"(ret) : "r"(a), "r"(b), "r"(c));
	return ret;
#else
	return (a<< b) + c;
#endif
}

DEVICE2 uint64 shl_add(uint a, uint b, uint64 c) {
	return (a<< b) + c;
}

// (a<< b) + c, where b is a constant. We hope to use the ISCADD instruction 
// rather than the vshl.add instruction.
DEVICE uint shl_add_c(uint a, uint b, uint c) {
	return (a<< b) + c;
}

DEVICE uint shr_add(uint a, uint b, uint c) {
#ifdef USE_VIDEO_INSTRUCTIONS
	uint ret;
	asm("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" : 
		"=r"(ret) : "r"(a), "r"(b), "r"(c));
	return ret;
#else
	return (a>> b) + c;
#endif
}

DEVICE uint mul_add(uint a, uint b, uint c) {
#ifdef USE_VIDEO_INSTRUCTIONS
	uint ret;
	asm("vmad.u32.u32.u32 %0, %1, %2, %3;" : 
		"=r"(ret) : "r"(a), "r"(b), "r"(c));
	return ret;
#else
	return (a * b) + c;
#endif
}

DEVICE uint imad(uint a, uint b, uint c) {
#ifdef USE_VIDEO_INSTRUCTIONS
	uint ret;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
	return ret;
#else
	return a * b + c;
#endif
}


DEVICE uint GetByte(uint a, uint i) {
	return prmt(a, 0, 0x4440 + i);
}
DEVICE uint ExpandUint8Low(uint a) {
	return prmt(a, 0, 0x4140);
}
DEVICE uint ExpandUint8High(uint a) {
	return prmt(a, 0, 0x4342);
}
DEVICE uint2 Expand8Uint4To8Uint8(uint a) {
	// b.x = (0xf & a) | // 0 -> 0
	// ((0xf0 & a) << 4) | // 4 -> 8
	// ((0xf00 & a) << 8) | // 8 -> 16
	// ((0xf000 & a) << 12); // 12 -> 24
	// b.y = ((0xf0000 & a) >> 16) | // 16 -> 0
	// ((0xf00000 & a) >> 12) | // 20 -> 8
	// ((0xf000000 & a) >> 8) | // 24 -> 16
	// ((0xf0000000 & a) >> 4); // 28 -> 24
	uint2 b;
	uint a2 = a>> 4;
	b.x = 0x0f0f0f0f & prmt(a, a2, 0x5140);
	b.y = 0x0f0f0f0f & prmt(a, a2, 0x7362);
	return b;
}

DEVICE uint StridedThreadOrder(uint index) {
	return index + (index / WARP_SIZE);
}


DEVICE uint LoadKey(const uint* keys_global, uint index, uint numElements, 
	bool checkRange) {
	uint key = 0xffffffff;
	if(checkRange)
		if(index < numElements) key = keys_global[index];
	else
		key = keys_global[index];
	return key;
}

// volatile qualifier appropriate for shared memory.
DEVICE2 uint LoadShifted(const volatile uint* shared, uint shiftedIndex) {
	return *((volatile uint*)(((volatile char*)shared) + shiftedIndex));
}
DEVICE2 void StoreShifted(volatile uint* shared, uint shiftedIndex, uint val) {
	*((volatile uint*)(((volatile char*)shared) + shiftedIndex)) = val;
}


// Put a float into radix order.
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

// Put a radix order into back into a float.
DEVICE uint FloatToUint(float f) {
	int x = __float_as_int(f);
	int bits = x>> 31;

	int flipped = x ^ (0x7fffffff & bits);

	int adjusted = 0x80000000 + flipped;

	uint u = (uint)adjusted;
	return u;
}

// Patches missing __hiloint2longlong function in CUDA library.
typedef volatile union {
	int64 ll;
	uint64 ull;
	double d;
	uint u[2];
} ConvertType64;

DEVICE2 uint64 __hilouint2ulonglong(uint2 x) {
	ConvertType64 t;
	t.u[0] = x.x;
	t.u[1] = x.y;
	return t.ull;
}

DEVICE2 uint2 __ulonglong2hilouint2(uint64 x) {
	ConvertType64 t;
	t.ull = x;
	return make_uint2(t.u[0], t.u[1]);	
}


////////////////////////////////////////////////////////////////////////////////

// Computes an iterator range from div_t result called on the host side.
DEVICE2 int2 ComputeTaskRange(int block, int taskQuot, int taskRem) {

	int2 range;
	range.x = taskQuot * block;
	range.x += min(block, taskRem);
	range.y = range.x + taskQuot + (block < taskRem);

	return range;
}

DEVICE2 int2 ComputeTaskRange(int block, int taskQuot, int taskRem, 
	int segSize, int count) {

	int2 range = ComputeTaskRange(block, taskQuot, taskRem);
	range.x *= segSize;
	range.y *= segSize;
	range.y = min(range.y, count);
	
	return range;
}



////////////////////////////////////////////////////////////////////////////////
// Perform an in-place scan over countScan_global. Scans 1<< NumBits elements
// starting at shared, using a single warp.

template<int NumBits>
DEVICE2 void IntraWarpParallelScan(uint tid, volatile uint* shared, 
	bool inclusive) {

	const int NumDigits = 1<< NumBits;

	if(NumBits <= 5) {
		if(tid < NumDigits) {
			uint x = shared[tid];
			uint sum = x;

			#pragma unroll
			for(int i = 0; i < NumBits; ++i) {
				uint offset = 1<< i;
				if(tid >= offset) {
					uint y = shared[tid - offset];
					x += y;
				}
				if(i < NumBits - 1) shared[tid] = x;
			}
			shared[tid] = inclusive ? (x - sum) : x;
		}
	} else if(6 == NumDigits) {
		if(tid < WARP_SIZE) {
			uint x0 = shared[tid];
			uint x1 = shared[WARP_SIZE + tid];

			uint sum0 = x0;
			uint sum1 = x1;

			#pragma unroll
			for(int i = 0; i < LOG_WARP_SIZE; ++i) {
				uint offset = 1<< i;
				if(tid >= offset) {
					uint y0 = shared[tid - offset];
					x0 += y0;
				}
				uint y1 = shared[WARP_SIZE + tid - offset];
				x1 += y1;
				if(LOG_WARP_SIZE - 1 == i) x1 += x0;
				else {
					shared[tid] = x0;
					shared[WARP_SIZE + tid] = x1;
				}
			}
			shared[tid] = inclusive ? (x0 - sum0) : x0;
			shared[WARP_SIZE + tid] = inclusive ? (x1 - sum1) : x1;
		}
	} else if(7 == NumDigits) {
		if(tid < WARP_SIZE) {
			uint x0 = shared[tid];
			uint x1 = shared[WARP_SIZE + tid];
			uint x2 = shared[2 * WARP_SIZE + tid];
			uint x3 = shared[3 * WARP_SIZE + tid];
			uint sum0 = x0;
			uint sum1 = x1;
			uint sum2 = x2;
			uint sum3 = x3;

			#pragma unroll
			for(int i = 0; i < LOG_WARP_SIZE; ++i) {
				uint offset = 1<< i;
				if(tid >= offset) {
					uint y0 = shared[tid - offset];
					x0 += y0;
				}
				uint y1 = shared[WARP_SIZE + tid - offset];
				uint y2 = shared[2 * WARP_SIZE + tid - offset];
				uint y3 = shared[3 * WARP_SIZE + tid - offset];

				x1 += y1;
				x2 += y2;
				x3 += y3;

				if(LOG_WARP_SIZE - 1 == i) {
					x1 += x0;
					x2 += x1;
					x3 += x2;
				} else {
					shared[tid] = x0;
					shared[WARP_SIZE + tid] = x1;
					shared[2 * WARP_SIZE + tid] = x2;
					shared[3 * WARP_SIZE + tid] = x3;
				}
			}
			shared[tid] = inclusive ? (x0 - sum0) : x0;
			shared[WARP_SIZE + tid] = inclusive ? (x1 - sum1) : x1;
			shared[2 * WARP_SIZE + tid] = inclusive ? (x2 - sum2) : x2;
			shared[3 * WARP_SIZE + tid] = inclusive ? (x3 - sum3) : x3;
		}
	}
}
