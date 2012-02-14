#pragma once

#include "common.cu"

// Include just once - this code is the same no matter NUM_BUCKETS

#define MAX_BITS 6
#define MAX_BUCKETS (1<< MAX_BITS)

typedef uint Values[VALUES_PER_THREAD];

#define BIT(bit) (1<< bit)
#define MASK(high, low) (((31 == high) ? 0 : (1u<< (high + 1))) - (1<< low))
#define MASKSHIFT(high, low, x) ((MASK(high, low) & x) >> low)


// #define MAX_TRANS(values, buckets) (5 * (values / WARP_SIZE + buckets) / 4)

////////////////////////////////////////////////////////////////////////////////
// Load global keys and values.

// LoadWarpValues loads so that each warp has values that are contiguous in
// global memory. Rather than having a stride of NUM_THREADS, the values are
// strided with WARP_SIZE. This is advantageous, as it lets us scatter to shared
// memory, the read into register arrays in thread-order, and process in 
// thread-order, without requiring __syncthreads, as the warps are independent.

// Read from global memory into warp order.
DEVICE void LoadWarpValues(const uint* values_global, uint warp, uint lane, 
	uint block, Values values) {

	uint threadStart = NUM_VALUES * block + 
		VALUES_PER_THREAD * WARP_SIZE * warp + lane;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = values_global[threadStart + v * WARP_SIZE];
}

// Read from global memory into block order.
DEVICE void LoadBlockValues(const uint* values_global, uint tid, uint block, 
	Values values) {

	uint threadStart = NUM_VALUES * block + tid;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = values_global[threadStart + v * NUM_THREADS];
}


////////////////////////////////////////////////////////////////////////////////
// Gather and scatter to shared memory.

// If strided is true, move keys/fused keys to shared memory with a 33 slot 
// stride between rows. This allows conflict-free gather into thread order.

DEVICE void GatherWarpOrder(uint warp, uint lane, bool strided, Values values,
	const uint* scattergather_shared) {

	// Each warp deposits VALUES_PER_THREAD rows to shared mem. The stride 
	// between rows is WARP_SIZE + 1. This enables shared loads with no bank
	// conflicts.
	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = scattergather_shared[sharedStart + stride * v];
}
DEVICE void ScatterWarpOrder(uint warp, uint lane, bool strided,
	const Values values, uint* scattergather_shared) {
	// Each warp deposits VALUES_PER_THREAD rows to shared mem. The stride
	// between rows is WARP_SIZE + 1. This enables shared loads with no bank
	// conflicts.
	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		scattergather_shared[sharedStart + stride * v] = values[v];
}


DEVICE void ScatterBlockOrder(uint tid, bool strided, uint numThreads,
	const Values values, uint* scattergather_shared) {

	uint* shared = scattergather_shared + tid;
	if(strided) shared += tid / WARP_SIZE;

	uint stride = strided ? (numThreads + numThreads / WARP_SIZE) : numThreads;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		shared[stride * v] = values[v];
}
DEVICE void GatherBlockOrder(uint tid, bool strided, uint numThreads,
	Values values, const uint* scattergather_shared) {

	const uint* shared = scattergather_shared + tid;
	if(strided) shared += tid / WARP_SIZE;

	uint stride = strided ? (numThreads + numThreads / WARP_SIZE) : numThreads;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = shared[stride * v];
}


DEVICE void GatherFromIndex(const Values gather, bool premultiplied, 
	Values data, const uint* scattergather_shared) {

	if(premultiplied) {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			data[v] = *(uint*)((char*)(scattergather_shared) + gather[v]);
	} else {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			data[v] = scattergather_shared[gather[v]];
	}
}

DEVICE void ScatterFromIndex(const Values scatter, bool premultiplied, 
	const Values data, uint* scattergather_shared) {

	if(premultiplied) {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			*(uint*)((char*)(scattergather_shared) + scatter[v]) = data[v];
	} else {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			scattergather_shared[scatter[v]] = data[v];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Fused key support

// Build fused keys by packing the radix digits into 31:24 and the index within
// the block into 23:0.

DEVICE void BuildFusedKeysWarpOrder(uint warp, uint lane, const Values keys,
	uint bitOffset, uint numBits, Values fusedKeys, bool strided) {

	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint key = bfe(keys[v], bitOffset, numBits);
		uint index = 4 * (sharedStart + stride * v);
		fusedKeys[v] = index + (key<< 24);
	}
}


DEVICE void BuildFusedKeysThreadOrder(uint tid, const Values keys, 
	uint bitOffset, uint numBits, Values fusedKeys, bool strided) {

//	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	// uint sharedStart = 

	// blHA

}

DEVICE void BuildGatherFromFusedKeys(const Values fusedKeys, Values scatter) {
	#pragma unroll 
	for(int v = 0; v < VALUES_PER_THREAD; ++v) 
		scatter[v] = 0x00ffffff & fusedKeys[v];
}
