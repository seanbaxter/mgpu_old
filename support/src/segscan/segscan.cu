// Demonstration of segmented scan. 
// See http://www.moderngpu.com/sparse/segscan.html

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define NUM_THREADS 256
#define NUM_WARPS 8
#define LOG_NUM_WARPS 3

#define VALUES_PER_THREAD 8

#define DEVICE extern "C" __device__ __forceinline__

#include <device_functions.h>

typedef unsigned int uint;

DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


////////////////////////////////////////////////////////////////////////////////
// Use ballot and clz to run a segmented scan over a single warp, with one value
// per thread.

extern "C" __global__ void SegScanWarp(const uint* dataIn_global,
	uint* dataOut_global) {

	uint tid = threadIdx.x;
	uint packed = dataIn_global[tid];

	// The start flag is in the high bit.
	uint flag = 0x80000000 & packed;

	// Get the start flags for each thread in the warp.
	uint flags = __ballot(flag);

	// Mask out the bits above the current thread.
	flags &= bfi(0, 0xffffffff, 0, tid + 1);

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	uint distance = __clz(flags) - tid;

	__shared__ volatile uint shared[WARP_SIZE];

	uint x = 0x7fffffff & packed;
	uint x2 = x;
	shared[tid] = x;

	// Perform the parallel scan. Note the conditional if(offset < distance)
	// replaces the ordinary scan conditional if(offset <= tid).
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(offset <= distance) x += shared[tid - offset];
		shared[tid] = x;
	}

	// Turn inclusive scan into exclusive scan.
	x -= x2;
		
	dataOut_global[tid] = x;
}


////////////////////////////////////////////////////////////////////////////////
// Use parallel scan to compute the ranges for a segmented scan over a warp with
// eight values per thread.

extern "C" __global__ void SegScanWarp8(const uint* dataIn_global,
	uint* dataOut_global) {

	uint tid = threadIdx.x;
	
	__shared__ volatile uint shared[VALUES_PER_THREAD * (WARP_SIZE + 1)];
	
	// Load packed values from global memory and scatter to shared memory. Use
	// a 33-slot stride between successive values in each thread to set us up
	// for a conflict-free strided order -> thread order transpose.

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint x = dataIn_global[i * WARP_SIZE + tid];
		shared[i * (WARP_SIZE + 1) + tid] = x;
	}

	uint offset = VALUES_PER_THREAD * tid;
	offset += offset / WARP_SIZE;
	uint packed[VALUES_PER_THREAD];

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i)
		packed[i] = shared[offset + i];


	////////////////////////////////////////////////////////////////////////////
	// UPSWEEP PASS
	// Run a sequential segmented scan for all values in the packed array. Find
	// the sum of all values in the thread's last segment. Additionally set
	// index to tid if any segments begin in this thread.
	
	uint last = 0;
	uint index = 0;

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint flag = 0x80000000 & packed[i];
		uint x = 0x7fffffff & packed[i];
		if(!flag) last = 0;
		else index = tid;
		last += x;
	}


	////////////////////////////////////////////////////////////////////////////
	// SEGMENT PASS
	// Run a max() scan to find the thread containing the start value for the 
	// segment that begins this thread.

	volatile int* indices = (volatile int*)shared;

	// Zero out the preceding 16 elements to allow a scan without conditionals.
	indices[tid] = 0;
	indices += WARP_SIZE / 2;
	indices[0] = index;

	index = 0;

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		uint left = indices[-offset];
		index = max(index, left);
		indices[0] = index;
	}


	////////////////////////////////////////////////////////////////////////////
	// REDUCE PASS
	// Run a prefix sum scan over last to compute for each thread the sum of all
	// values in the segmented preceding the current thread, up to that point.
	// This is added back into the thread-local exclusive scan for the continued
	// segment in each thread.

	shared[tid] = last;
	uint first = shared[tid - index];
	uint sum = 0;

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		if(tid - offset > index) {
			uint left = shared[tid - offset];
			sum += left;
		}
		shared[tid] = sum;
	}
	if(index < tid) sum += first;


	////////////////////////////////////////////////////////////////////////////
	// DOWNSWEEP PASS
	// Add sum to all the values in the continuing segment (that is, before the
	// first start flag) in this thread.

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		if(0x80000000 & packed[i]) sum = 0;
		uint x = 0x7fffffff & packed[i];

		shared[offset + i] = sum;
		sum += x;
	}


	// Store the values back to global memory.
	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint x = shared[i * (WARP_SIZE + 1) + tid];
		dataOut_global[i * WARP_SIZE + tid] = x;
	}
}

extern "C" __global__ void SegScanBlock8(const uint* dataIn_global,
	uint* dataOut_global) {


}