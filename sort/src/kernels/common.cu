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
// Perform a parallel scan over 1<< Levels lanes in the warp. The calling code
// should branch so that only the 1<< Levels threads are active. shared must
// be large enough so that every active thread can store its value. 

template<int Levels>
DEVICE2 uint IntraWarpParallelScan(uint lane, uint x, volatile uint* shared,
	bool noPred, bool inc) {

	const int Count = 1<< Levels;
	const int Prefix = Count / 2;
	if(noPred) {
		shared[lane] = 0;
		shared += Prefix;
	}

	uint sum = x;
	shared += lane;
	shared[0] = x;

	#pragma unroll
	for(int i = 0; i < Levels; ++i) {
		uint offset = 1<< i;
		if(noPred)
			x += shared[-offset];
		else if(lane >= offset)
			x += shared[-offset];
		if(i < Levels - 1)
			shared[0] = x;
	}
	if(!inc) x -= sum;

	return x;
}

// Run a scan over 64 elements where each thread manages two values. Adjacent
// values are loaded into threads to allow for a single parallel scan.

// 
// If type == 0, val.x is 2 * lane + 0 and val.y is 2 * lane + 1 (optimal).
// If type == 1, val.x is lane and val.y is lane + 32.
DEVICE2 uint2 IntraWarpScan64(uint lane, uint2 val, volatile uint* shared,
	bool noPred, bool inc, int type) {

	if(1 == type) {
		// Put into type 0 order.
		shared[lane] = val.x;
		shared[WARP_SIZE + 1 + lane] = val.y;
		
		volatile uint* start = shared + (lane > WARP_SIZE / 2);
		val.x = start[2 * lane];
		val.y = start[2 * lane + 1];
	}

	uint sum = val.x + val.y;
	uint scan = IntraWarpParallelScan<LOG_WARP_SIZE>(lane, sum, shared, noPred,
		inc);

	if(inc) {
		val.x = scan - val.y;
		val.y = scan;
	} else {
		val.y = scan + val.x;
		val.x = scan;
	}

	if(1 == type) {
		volatile uint* start = shared + (lane / (WARP_SIZE / 2));
		start[2 * lane] = val.x;
		start[2 * lane + 1] = val.y;

		val.x = shared[lane];
		val.y = shared[WARP_SIZE + 1 + lane];
	}
	return val;
}

DEVICE2 uint4 IntraWarpScan128(uint lane, uint4 val, volatile uint* shared,
	bool noPred, bool inc, int type) {

	if(1 == type) {
		// Put into type 0 order.
		shared[lane] = val.x;
		shared[WARP_SIZE + 1 + lane] = val.y;
		shared[2 * WARP_SIZE + 2 + lane] = val.z;
		shared[3 * WARP_SIZE + 3 + lane] = val.w;
		
		volatile uint* start = shared + (lane / (WARP_SIZE / 4));
		val.x = start[4 * lane];
		val.y = start[4 * lane + 1];
		val.z = start[4 * lane + 2];
		val.w = start[4 * lane + 3];
	}

	uint offset1 = val.x + val.y;
	uint offset2 = offset1 + val.z;
	uint sum = offset2 + val.w;
	uint scan = IntraWarpParallelScan<LOG_WARP_SIZE>(lane, sum, shared, noPred,
		inc);

	if(inc) {
		val.x = scan - offset2;
		val.y = scan - offset1;
		val.z = scan - val.w;
		val.w = scan;
	} else {
		val.x = scan;
		val.y = scan + val.x;
		val.z = scan + offset1;
		val.w = scan + offset2;
	}

	if(1 == type) {
		volatile uint* start = shared + (lane / (WARP_SIZE / 4));
		start[4 * lane] = val.x;
		start[4 * lane + 1] = val.y;
		start[4 * lane + 2] = val.z;
		start[4 * lane + 3] = val.w;

		val.x = shared[lane];
		val.y = shared[WARP_SIZE + 1 + lane];
		val.z = shared[2 * WARP_SIZE + 2 + lane];
		val.w = shared[3 * WARP_SIZE + 3 + lane];
	}
	return val;
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
				if(tid >= offset)
					x += shared[tid - offset];
				if(i < NumBits - 1) shared[tid] = x;
			}
			shared[tid] = inclusive ? x : (x - sum);
		}
	} else if(6 == NumBits) {
		if(tid < WARP_SIZE) {
			uint x0 = shared[tid];
			uint x1 = shared[WARP_SIZE + tid];

			uint sum0 = x0;
			uint sum1 = x1;

			#pragma unroll
			for(int i = 0; i < LOG_WARP_SIZE; ++i) {
				uint offset = 1<< i;
				if(tid >= offset)
					x0 += shared[tid - offset];
				x1 += shared[WARP_SIZE + tid - offset];
				
				if(LOG_WARP_SIZE - 1 == i) x1 += x0;
				else {
					shared[tid] = x0;
					shared[WARP_SIZE + tid] = x1;
				}
			}
			shared[tid] = inclusive ? x0 : (x0 - sum0);
			shared[WARP_SIZE + tid] = inclusive ? x1 : (x1 - sum1);
		}
	} else if(7 == NumBits) {
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
				if(tid >= offset)
					x0 += shared[tid - offset];
				x1 += shared[WARP_SIZE + tid - offset];
				x2 += shared[2 * WARP_SIZE + tid - offset];
				x3 += shared[3 * WARP_SIZE + tid - offset];

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
			shared[tid] = inclusive ? x0 : (x0 - sum0);
			shared[WARP_SIZE + tid] = inclusive ? x1 : (x1 - sum1);
			shared[2 * WARP_SIZE + tid] = inclusive ? x2 : (x2 - sum2);
			shared[3 * WARP_SIZE + tid] = inclusive ? x3 : (x3 - sum3);
		}
	}
}
