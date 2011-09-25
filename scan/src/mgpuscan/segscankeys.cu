#define NUM_THREADS KEYS_NUM_THREADS
#define BLOCKS_PER_SM KEYS_BLOCKS_PER_SM
#define VALUES_PER_THREAD KEYS_VALUES_PER_THREAD

#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS LOG_BASE_2(NUM_WARPS)
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)


////////////////////////////////////////////////////////////////////////////////
// UPSWEEP PASS. Find the sum of all values in the last segment in each block.
// When the first head flag in the block is encountered, write out the sum to 
// that point and return. We only need to reduce the last segment to feed sums
// up to the reduction pass.

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepKeys(const uint* valuesIn_global, const uint* keysIn_global,
	uint* blockLast_global, int* headFlagPos_global,
	const int2* rangePairs_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;

	int2 range = rangePairs_global[block];

	const int UpsweepValues = 8;
	const int NumValues = UpsweepValues * NUM_THREADS;

	// Start at the last tile (NUM_VALUES before the end iterator). Because
	// upsweep isn't executed for the last block, we don't have to worry about
	// the ending edge case.
	int current = range.y - NumValues;

	uint threadSum = 0;
	uint blockFlags = 0;

	// Load the last key in the segment.
	uint lastKey = keysIn_global[range.y - 1];

	while(current >= range.x) {

		uint keys[UpsweepValues];
		uint x[UpsweepValues];

		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) {
			x[i] = valuesIn_global[current + tid + i * NUM_THREADS];
			keys[i] = keysIn_global[current + tid + i * NUM_THREADS];
		}

		// Add up all the values with a key that matches lastKey. If this thread
		// has any key that doesn't match lastKey, mark the prevSeg flag.
		bool prevSeg = false;

		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) {
			if(keys[i] == lastKey) threadSum += x[i];
			else prevSeg = true;
		}

		// Use ballot to see if any threads in this warp encountered an earlier
		// segment.
		uint warpFlags = __ballot(prevSeg);

		__shared__ volatile uint warpShared[NUM_WARPS];
		if(!lane) warpShared[warp] = warpFlags;
		__syncthreads();

		if(tid < NUM_WARPS) {
			warpFlags = warpShared[tid];
			warpFlags = __ballot(warpFlags);
			warpShared[tid] = warpFlags;
		}
		__syncthreads();

		uint blockFlags = warpShared[0];

		if(blockFlags) break;

		current -= NumValues;
	}

	// We've either hit the preceding segment or run out of values. Do a
	// horizontal sum of the thread values and store to global memory.
	uint total = (uint)Reduce<NUM_WARPS>(tid, (int)threadSum, 0);

	if(0 == tid) {
		blockLast_global[block] = total;

		// Prepare the head flag.
		uint headFlag = blockFlags;
		if(!headFlag && range.x) {
			// Load the preceding key.
			uint precedingKey = keysIn_global[range.x - 1];
			headFlag = precedingKey != lastKey;
		}
		headFlagPos_global[block] = headFlag;
	}
}


////////////////////////////////////////////////////////////////////////////////
// DOWNSWEEP PASS.

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepKeys(const uint* valuesIn_global, 
	const uint* keysIn_global, uint* valuesOut_global, const uint* start_global,
	const int2* rangePairs_global, int count, int inclusive) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint index = VALUES_PER_WARP * warp + lane;

	int2 range = rangePairs_global[block];

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];
	__shared__ volatile uint blockOffset_shared;

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Transpose values into thread order.
	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;

	int lastOffset = ~(NUM_VALUES - 1) & count;

	if(!tid) blockOffset_shared = start_global[block];

	__shared__ volatile uint precedingKey_shared;
	if(!tid)
		precedingKey_shared = block ? keysIn_global[range.x - 1] : 0;
	
	while(range.x < range.y) {

		// Load values into packed.
		uint x[VALUES_PER_THREAD];
		uint keys[VALUES_PER_THREAD];

		////////////////////////////////////////////////////////////////////////
		// Load and transpose values.
		
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint source = range.x + index + i * WARP_SIZE;
				uint x = valuesIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = x;
			}

		// Transpose into thread order 
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			x[i] = warpShared[offset + i];


		////////////////////////////////////////////////////////////////////////
		// Load and transpose keys.

			
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint source = range.x + index + i * WARP_SIZE;
				uint key = keysIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = key;
			}

		// Transpose into thread order 
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			keys[i] = warpShared[offset + i];

		__syncthreads();

		// Store the last key for each thread in shared memory.
		shared[1 + tid] = keys[VALUES_PER_THREAD - 1];
		__syncthreads();

		// Retrieve the last key for the preceding thread.
		uint precedingKey = shared[tid];
		if(!tid) {
			precedingKey = precedingKey_shared;
			precedingKey_shared = shared[NUM_THREADS];
		}
	
		////////////////////////////////////////////////////////////////////////
		// Compare the adjacent keys in each thread to derive head flags.

		uint flags[VALUES_PER_THREAD];

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			if(i) flags[i] = keys[i - 1] != keys[i];
			else flags[0] = keys[0] != precedingKey;
		}
		

		////////////////////////////////////////////////////////////////////////
		// Run downsweep function on values and head flags.

		SegScanDownsweep<NUM_WARPS, VALUES_PER_THREAD>(tid, lane, warp, x, 
			flags, warpShared, threadShared, inclusive, &blockOffset_shared);

		// Transpose 
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			warpShared[offset + i] = x[i];

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint target = range.x + index + i * WARP_SIZE;
			valuesOut_global[target] = threadShared[i * (WARP_SIZE + 1)];
		}

		range.x += NUM_VALUES;
	}
}


#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES
