#pragma once

#include "sortcommon.cu"

////////////////////////////////////////////////////////////////////////////////
// SortLocal

// Given keys in thread order (fusedKeys) or keys in shared memory in strided
// order (scattergather_shared), sort between 1 and 7 key bits and store into
// shared memory.

template<int NumThreads, int NumBits>
DEVICE2 void SortLocal(uint tid, Values fusedKeys, uint bit, bool loadFromArray,
	uint* scattergather_shared, uint* scratch_shared, uint* debug_global) {

	if(1 == NumBits) 
		SortAndScatter(tid, fusedKeys, bit, 1, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(2 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(3 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(4 == NumBits) {
		/*SortAndScatter(tid, fusedKeys, bit, 4, NumThreads,
			!LoadFromTexture, false, scattergather_shared, scratch_shared, 
			debug_global_out);*/
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(5 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(6 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 3, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(7 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, true,
			scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 4, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	}
}


////////////////////////////////////////////////////////////////////////////////

template<int NumThreads, int NumBits, int ValueCount, bool UseScatterList,
	bool LoadFromTexture, bool DetectEarlyExit>
DEVICE2 void SortFunc(const uint* keys_global_in, uint firstBlock,
	const uint* bucketCodes_global, uint bit, uint* keys_global_out,
	uint numValueStreams, uint* debug_global_out, 
	const uint* values1_global_in, const uint* values2_global_in,
	const uint* values3_global_in, const uint* values4_global_in,
	const uint* values5_global_in, const uint* values6_global_in,
	// For VALUE_TYPE_MULTI, we have to pass each of the pointers in as 
	// individual arguments, not as arrays, or CUDA generates much worse code,
	// full of unified addressing instructions.
	uint* values1_global_out, uint* values2_global_out,
	uint* values3_global_out, uint* values4_global_out,
	uint* values5_global_out, uint* values6_global_out) {

	const int NumValues = VALUES_PER_THREAD * NumThreads;
	const int NumWarps = NumThreads / WARP_SIZE;
	const int Stride = WARP_SIZE + 1;

	const int NumBuckets = 1<< NumBits;

	// Simple scatter
	const int ScratchSize = 2 * (NumThreads + NumWarps) + 4 * WARP_SIZE + 32;

	__shared__ uint scratch_shared[ScratchSize];

	const int ScatterStructSize = NumBuckets;
	
	__shared__ uint scatterList_shared[ScatterStructSize];
	__shared__ uint scattergather_shared[NumWarps * VALUES_PER_THREAD * Stride];


	////////////////////////////////////////////////////////////////////////////
	// LOAD KEYS, CREATE FUSED KEYS, AND REINDEX INTO THREAD ORDER

	uint tid = threadIdx.x;
	uint block = blockIdx.x + firstBlock;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	debug_global_out += NumValues * block;

	// Load the scatter (transaction) structure.
	uint globalStructOffset = ScatterStructSize * block;
	if(tid < ScatterStructSize)
		scatterList_shared[tid] = 
			bucketCodes_global[globalStructOffset + tid];

	// Load the keys and, if sorting values, create fused keys. Store into 
	// shared mem with a WARP_SIZE + 1 stride between warp rows, so that loads
	// into thread order occur without bank conflicts.
	Values keys, fusedKeys;

	if(LoadFromTexture) {
		// Load keys from a texture. The texture sampler serves as an 
		// asynchronous independent subsystem. It helps transpose data from
		// strided to thread order without involving the shader units.

		uint keysOffset = NumValues * blockIdx.x + VALUES_PER_THREAD * tid;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
			uint4 k = tex1Dfetch(keys_texture_in, keysOffset / 4 + i);
			keys[4 * i + 0] = k.x;
			keys[4 * i + 1] = k.y;
			keys[4 * i + 2] = k.z;
			keys[4 * i + 3] = k.w;
		}

		if(0 == ValueCount)
			// Sort only keys.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i)
				fusedKeys[i] = keys[i];
		else
			// Sort key-value tuples.
			BuildFusedKeysThreadOrder(tid, keys, bit, NumBits, fusedKeys,
				false);

	} else {
		// Load keys from global memory. This requires using shared memory to 
		// transpose data from strided to thread order.

		LoadWarpValues(keys_global_in, warp, lane, block, keys);

		if(0 == ValueCount)
			// Sort only keys.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i)
				fusedKeys[i] = keys[i];
		else
			// Sort key-value tuples.
			BuildFusedKeysWarpOrder(warp, lane, keys, bit, NumBits, fusedKeys,
				false);

		// Store the keys or fused keys into shared memory for the
		// strided->thread order transpose.
		ScatterWarpOrder(warp, lane, true, fusedKeys, scattergather_shared);
	}


	////////////////////////////////////////////////////////////////////////////
	// Check the early exit code for this block.

	bool isEarlyDetect = false;
/*	if(DetectEarlyExit) {
		if(!tid) {
			uint scatter = scatterList_shared[0];
			scratch_shared[0] = 1 & scatter;
			scatterList_shared[0] = ~1 & scatter;		
		}
		__syncthreads();
		isEarlyDetect = scratch_shared[0];
	}*/

	// Sort the fused keys in shared memory if early exit was not detected.
	if(!isEarlyDetect) {
		uint scanBitOffset = ValueCount ? 24 : bit;

		SortLocal<NumThreads, NumBits>(tid, fusedKeys, scanBitOffset,
			!LoadFromTexture, scattergather_shared, scratch_shared, 
			debug_global_out);
	} else if(LoadFromTexture) {
		// Copy the data fusedKeys to shared memory?


	}


	////////////////////////////////////////////////////////////////////////////
	// Store the keys and values to global memory.

	if(0 == ValueCount) {


	//	#pragma unroll
	//	for(int v = 0; v < VALUES_PER_THREAD; ++v)
	//		keys_global_out[NumValues * block + NumThreads * v + tid] =
	//			scattergather_shared[NumThreads * v + tid];

		// Store only keys.
		GatherBlockOrder(tid, false, NumThreads, keys, scattergather_shared);
		ScatterKeysSimple(tid, keys_global_out, bit, NumBits, 
			scatterList_shared, keys);

	} else if(-1 == ValueCount) {
		// Store keys and indices.
/*
		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		Values gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);

		// Generate index values from the gather indices. As these are 
		// pre-multiplied, divide them by 4 and add the block offset.
		uint blockOffset = NUM_VALUES * block;
		Values indices;
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			indices[v] = shr_add(gather[v], 2, blockOffset);

		ScatterPairSimple(tid, keys_global_out, index_global_out, bit, NUM_BITS,
			(const uint*)compressedList_shared, keys, indices);*/

	} else if(1 == ValueCount) {
		// Store key-value pairs.

		/*
		Values values;
		LoadBlockValues(value1_global_in, tid, block, values);
	
		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		Values gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);
		__syncthreads();

		// Store the values to shared memory.
		ScatterBlockOrder(tid, false, values);
		// ScatterWarpOrder(warp, lane, false, values);
		__syncthreads();

		// Gather the values from shared memory.
		GatherFromIndex(gather, true, values);

		ScatterPairSimple(tid, keys_global_out, value1_global_out, bit, 
			NUM_BITS, (const uint*)compressedList_shared, keys, values);
*/
	} else {
		// Store keys with multiple value streams.

		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		/*
		Values gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);
		__syncthreads();

		Values globalScatter;
		MultiScatterSimple(tid, keys_global_out, bit, NUM_BITS,
			(const uint*)compressedList_shared, keys, globalScatter);

		GlobalGatherScatter(tid, block, values1_global_in, 
			values1_global_out, gather, globalScatter);

		GlobalGatherScatter(tid, block, values2_global_in, 
			values2_global_out, gather, globalScatter);

		if(numValueStreams >= 3)
			GlobalGatherScatter(tid, block, values3_global_in, 
				values3_global_out, gather, globalScatter);

		if(numValueStreams >= 4)
			GlobalGatherScatter(tid, block, values4_global_in, 
				values4_global_out, gather, globalScatter);

		if(numValueStreams >= 5)
			GlobalGatherScatter(tid, block, values5_global_in, 
				values5_global_out, gather, globalScatter);

		if(6 == numValueStreams)
			GlobalGatherScatter(tid, block, values6_global_in, 
				values6_global_out, gather, globalScatter);
*/
	}


}

/*
#define GEN_SORT_FUNC(Name, NumThreads, NumBits, ValueCount,				\
	UseScatterList, LoadFromTexture, EarlyExit, BlocksPerSM)				\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global_in, uint firstBlock,						\
	const uint* bucketCodes_global, uint bit, uint* keys_global_out,		\
	uint numValueStreams, uint* debug_global_out,							\
	const uint* values1_global_in, const uint* values2_global_in,			\
	const uint* values3_global_in, const uint* values4_global_in,			\
	const uint* values5_global_in, const uint* values6_global_in,			\
	uint* values1_global_out, uint* values2_global_out,						\
	uint* values3_global_out, uint* values4_global_out,						\
	uint* values5_global_out, uint* values6_global_out) {					\
																			\
	SortFunc<NumThreads, NumBits, ValueCount, UseScatterList,				\
		LoadFromTexture, EarlyExit>(										\
		keys_global_in, firstBlock, bucketCodes_global, bit,				\
		keys_global_out, numValueStreams, debug_global_out,					\
		values1_global_in, values2_global_in, values3_global_in,			\
		values4_global_in, values5_global_in, values6_global_in,			\
		values1_global_out, values2_global_out, values3_global_out,			\
		values4_global_out, values5_global_out, values6_global_out);		\
}*/

#define GEN_SORT_FUNC(Name, NumThreads, NumBits, ValueCount,				\
	UseScatterList, LoadFromTexture, EarlyExit, BlocksPerSM)				\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global_in, uint firstBlock,						\
	const uint* bucketCodes_global, uint bit, uint* keys_global_out) {		\
																			\
	SortFunc<NumThreads, NumBits, ValueCount, UseScatterList,				\
		LoadFromTexture, EarlyExit>(										\
		keys_global_in, firstBlock, bucketCodes_global, bit,				\
		keys_global_out, 0, keys_global_out,								\
		0, 0, 0, 0, 0, 0,													\
		0, 0, 0, 0, 0, 0);													\
}




//GEN_SORT_FUNC(RadixSort_1, NUM_THREADS, 1, VALUE_COUNT, false,			\
//	LOAD_FROM_TEXTURE, false, NUM_BLOCKS)







/*
#define GEN_SORT_FUNC_

#ifdef VALUE_TYPE_NONE
void SORT_FUNC(const uint* keys_global_in, uint firstBlock,
	const uint* bucketCodes_global, uint bit, uint* keys_global_out) {

#elif defined(VALUE_TYPE_INDEX)
void SORT_FUNC(const uint* keys_global_in, uint firstBlock,
	const uint* bucketCodes_global, uint bit, uint* keys_global_out,
	uint* index_global_out) {

#elif defined(VALUE_TYPE_SINGLE)
void SORT_FUNC(const uint* keys_global_in, uint firstBlock,
	const uint* bucketCodes_global, uint bit, uint* keys_global_out, 
	const uint* value1_global_in, uint* value1_global_out) {

#elif defined(VALUE_TYPE_MULTI)
	// For VALUE_TYPE_MULTI, we have to pass each of the pointers in as 
	// individual arguments, not as arrays, or CUDA generates much worse code,
	// full of unified addressing instructions.
void SORT_FUNC(const uint* keys_global_in, uint firstBlock,
	const uint* bucketCodes_global, uint bit, uint* keys_global_out,
	uint numValueStreams,
//	const uint* values_global_in[6], uint* values_global_out[6]
	const uint* values1_global_in, const uint* values2_global_in,
	const uint* values3_global_in, const uint* values4_global_in,
	const uint* values5_global_in, const uint* values6_global_in,
	uint* values1_global_out, uint* values2_global_out,
	uint* values3_global_out, uint* values4_global_out,
	uint* values5_global_out, uint* values6_global_out) {
	*/