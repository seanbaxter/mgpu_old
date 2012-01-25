

// Segregate the fused keys bit-by-bit to sort. When this process is completed,
// read the fused keys back into registers. Bits 6-18 of each fused key is 
// the gather index for retrieving keys. It is also used for fetching values
// from global memory.

// There are three parallel scan types:
// 1 bit scan - count the number of keys with the sort bit set. this gives us 
//   the population of bucket 0. the population of bucket 1 is found by 
//   inference.
// 2 bit scan - count the populations of four buckets. Pack into 4uint8 DWORD.
//   Perform intra-warp parallel scan on these DWORDs. Unpack into 4uint16 
//   (DWORD pair) for inter-warp multiscan phase
// 3 bit scan - count the populations of eight buckets. Pack into 8uint8 DWORD
//   pair. Perform intra-warp parallel scan on these DWORD pairs. Unpack into
//   8uint16 (DWORD quad) for inter-warp multiscan phase.


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

	const int NumWarps = NumThreads / WARP_SIZE;
	const int Stride = LoadFromTexture ? WARP_SIZE : (WARP_SIZE + 1);

	const int NumBuckets = 1<< NumBits;

	// Reserve enough scratch space for the scans. The 3-bit multi-scan requires
	// the most shared memory. It stores 2 values per thread. These are 
	// strided with an extra slot for each WARP_SIZE of elements. Additionally,
	// 64 slots are required to hold the sequential scan results. Add in another
	// 32 slots for scan offsets.
	const int ScratchSize = 2 * (NumThreads + (NumThreads / WARP_SIZE)) + 96;

	__shared__ uint scratch_shared[ScratchSize];


	// Simple scatter
	const int ScatterStructSize = NumBuckets;
	
	__shared__ uint scatterList_shared[ScatterStructSize];
	__shared__ uint scattergather_shared[NumWarps * VALUES_PER_THREAD * Stride];


	////////////////////////////////////////////////////////////////////////////
	// LOAD KEYS, CREATE FUSED KEYS, AND REINDEX INTO THREAD ORDER

	uint tid = threadIdx.x;
	uint block = blockIdx.x + firstBlock;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	debug_global_out += NUM_VALUES * block;

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

		uint keysOffset = NUM_VALUES * blockIdx.x + VALUES_PER_THREAD * tid;

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
	if(DetectEarlyExit) {
		__shared__ int isEarlyExit_shared;
		if(!tid) {
			uint scatter = scatterList_shared[0];
			isEarlyExit_shared = 1 & scatter;
			scatterList_shared[0] = ~1 & scatter;		
		}
		__syncthreads();
		isEarlyDetect = isEarlyExit_shared;
	}

	// Sort the fused keys in shared memory if early exit was not detected.
	if(!isEarlyDetect) {
		uint scanBitOffset = ValueCount ? 24 : bit;

		if(1 == NumBits) 
			SortAndScatter(tid, fusedKeys, scanBitOffset, 1, !LoadFromTexture,
				false, debug_global_out);
		else if(2 == NumBits)
			SortAndScatter(tid, fusedKeys, scanBitOffset, 2, !LoadFromTexture,
				false, debug_global_out);
		else if(3 == NumBits)
			SortAndScatter(tid, fusedKeys, scanBitOffset, 3, !LoadFromTexture,
				false, debug_global_out);
		else if(4 == NumBits) {
			SortAndScatter(tid, fusedKeys, scanBitOffset, 2, !LoadFromTexture,
				true, debug_global_out);
			SortAndScatter(tid, fusedKeys, scanBitOffset + 2, 2, true,
				false, debug_global_out);
		} else if(5 == NumBits) {
			SortAndScatter(tid, fusedKeys, scanBitOffset, 2, !LoadFromTexture,
				true, debug_global_out);
			SortAndScatter(tid, fusedKeys, scanBitOffset + 2, 3, true,
				false, debug_global_out);
		} else if(6 == NumBits) {
			SortAndScatter(tid, fusedKeys, scanBitOffset, 3, !LoadFromTexture,
				true, debug_global_out);
			SortAndScatter(tid, fusedKeys, scanBitOffset + 3, 3, true,
				false, debug_global_out);
		}
	}


	////////////////////////////////////////////////////////////////////////////
	// Store the keys and values to global memory.

	if(0 == ValueCount) {
		// Store only keys.
		GatherBlockOrder(tid, false, keys);
		ScatterKeysSimple(tid, keys_global_out, bit, NumBits, 
			scattergather_shared, keys);
	
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
}

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