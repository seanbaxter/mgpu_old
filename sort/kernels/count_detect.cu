
// Have each warp process a histogram for one block. If the block has NUM_VALUES,
// each thread process NUM_VALUES / WARP_SIZE. Eg if NUM_VALUES is 2048, each 
// thread process 64 values. This operation is safe for NUM_VALUES up to 4096.

#define NUM_BUCKETS (1<< NUM_BITS)
#define NUM_COUNTERS ((NUM_BUCKETS + 3) / 4)
#define NUM_CHANNELS ((NUM_BUCKETS + 1) / 2)

#define PACKED_SPACING ((1 == NUM_BITS) ? 16 : 8)

// Only reserve as much shared memory as required - NUM_COUNTERS ints per thread.
// For 6 == NUM_BITS, this is 16 ints per thread, for 16384 bytes for a 256 thread
// group, allowing 50% occupancy. All smaller keys give 83% occupancy, as they
// are register limited.

#define SHARED_WARP_MEM MAX(WARP_SIZE * NUM_COUNTERS, WARP_SIZE * 2)

#define counts_shared COUNT_SHARED_MEM

// To detect data that is already sorted, reserve an additional 64 ints per warp.
__shared__ volatile uint COUNT_SHARED_MEM[NUM_WARPS * (SHARED_WARP_MEM + 64)];

// sorted_shared is scratch space for testing if each segment of loaded keys are
// sorted.
#define sorted_shared (counts_shared + NUM_WARPS * SHARED_WARP_MEM)


// To sorted_global, write the first key encountered and the last key encountered in the 
// count block if all all values in the count block are sorted. If the values aren't all
// sorted, don't spend the transaction writing to the array.

// NOTE: for 6 == NUM_BITS, high shared memory usage allows only 33% occupancy, when
// using DETECT_SORTED. It may be possible to instead run the inner loop,
// then back up two rows of accumulate, and test for sortedness.
// More work may be required.

#define INNER_LOOP 4

extern "C" __global__ __launch_bounds__(NUM_THREADS, 5) 
void COUNT_FUNC(const uint4* keys_global, uint bit, 
	uint numElements, uint valuesPerThread, uint* counts_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint warpStart = (block * NUM_WARPS + warp) * (WARP_SIZE * valuesPerThread / 4);

	volatile uint* warpCounters = counts_shared + warp * SHARED_WARP_MEM;
	volatile uint* counters = warpCounters + lane;

	volatile uint* sorted_scratch = sorted_shared + 64 * warp + lane;
	int radixSortedCount = 0;
	int fullSortedCount = 0;
	uint firstValue;
	uint lastValue;

	// Define the counters so we can pass them to IncBucketCounter. They don't actually
	// get used unless NUM_BITS <= 3 however.
	uint counter0 = 0;
	uint counter1 = 0;
	
#if NUM_BITS > 3
	// clear all the counters
	#pragma unroll
	for(uint i = 0; i < NUM_COUNTERS; ++i)
		counters[WARP_SIZE * i] = 0;
#endif

	uint4 valuesVec[INNER_LOOP];
	if(warpStart >= numElements) {
		// Set the last bucket with all the values.
		counters[(NUM_COUNTERS - 1) * WARP_SIZE] = valuesPerThread<< (32 - PACKED_SPACING);

		radixSortedCount = valuesPerThread;
		fullSortedCount = valuesPerThread;
		firstValue = 0xffffffff;
		valuesVec[INNER_LOOP - 1].w = 0xffffffff;

	} else {

		// Unroll to read 8 values at a time
		const uint4* warpData = keys_global + warpStart + lane;

		// Clear the sorted_scratch space to avoid false failure on the first values.
		sorted_scratch[WARP_SIZE] = 0;

		uint end = valuesPerThread / (4 * INNER_LOOP);

		for(int i = 0; i < end; ++i) {

			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j)
				valuesVec[j] = warpData[j * WARP_SIZE];
			warpData += INNER_LOOP * WARP_SIZE;

			if(!i) firstValue = valuesVec[0].x;
			
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				uint4 buckets = ReadBuckets(valuesVec[j], bit, NUM_BITS);
				IncVecCounter(buckets, counters, counter0, counter1, NUM_BITS);
				CheckSorted(valuesVec[j], buckets, bit, NUM_BITS, sorted_scratch,
					radixSortedCount, fullSortedCount, lane, j);

				sorted_scratch[0] = radixSortedCount + fullSortedCount;
			}
		}

		// Write the counters to shared memory if they were stored in register.
	#if NUM_BITS <= 2
		counters[0] = counter0;
	#elif 3 == NUM_BITS
		counters[0] = counter0;
		counters[WARP_SIZE] = counter1;
	#endif
	}
	lastValue = valuesVec[INNER_LOOP - 1].w;

	// If there are multiple counters, run GatherSums until we only have one
	// row of counters remaining.
#if 1 == NUM_BITS
	// 16-bit packing was used from the start, so we go directly to parallel reduction.
#elif 2 == NUM_BITS
	// Grab even and odd counters, and add and widen
	uint a = warpCounters[~1 & lane];
	uint b = warpCounters[1 | lane];
	uint gather = (1 & lane) ? 0x4342 : 0x4140;
	uint sum = prmt(a, 0, gather) + prmt(b, 0, gather);
	counters[0] = sum;
#elif 3 == NUM_BITS
	// At this point we have to manually unroll the GatherSums calls, because nvcc
	// is stupid and complains "Advisory: Loop was not unrolled, not an innermost
	// loop." This is due to the unrolled loop in GatherSums.
	GatherSums<NUM_COUNTERS>(lane, GATHER_SUM_MODE, warpCounters);
	GatherSums<NUM_COUNTERS>(lane, 0, warpCounters);
#elif 4 == NUM_BITS
	GatherSums<NUM_COUNTERS>(lane, GATHER_SUM_MODE, warpCounters);
	GatherSums<NUM_COUNTERS>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 2>(lane, 0, warpCounters);
#elif 5 == NUM_BITS
	GatherSums<NUM_COUNTERS>(lane, GATHER_SUM_MODE, warpCounters);
	GatherSums<NUM_COUNTERS>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 2>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 4>(lane, 0, warpCounters);
#elif 6 == NUM_BITS
	GatherSums<NUM_COUNTERS>(lane, GATHER_SUM_MODE, warpCounters);
	GatherSums<NUM_COUNTERS>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 2>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 4>(lane, 0, warpCounters);
	GatherSums<NUM_COUNTERS / 8>(lane, 0, warpCounters);
#endif

	// There are probably still multiple copies of each sum. Perform a parallel scan
	// to add them all up.
	if(NUM_CHANNELS < WARP_SIZE) {
		volatile uint* reduction = warpCounters + lane;
		reduction[WARP_SIZE] = 0;
		uint x = reduction[0];
		#pragma unroll
		for(int i = 0; i < 6 - NUM_BITS; ++i) {
			uint offset = NUM_CHANNELS<< i;
			uint y = reduction[offset];
			x += y;
			reduction[0] = x;
		}
	}

	
	// Use ballot to check if each sort block is internally sorted.
	int fullSorted = __all(valuesPerThread == fullSortedCount);
	int radixSorted = __all(valuesPerThread == radixSortedCount);
 
	// Counters from each sort block will be accessed by different threads
	// in the first warp. To eliminate bank conflicts, space the values out
	// by adding the warp counter to sorted_scratch for each thread.

	// The last value of the last thread in the warp is at
	// sorted_scratch[warp + 3 + WARP_SIZE - 1], i.e.
	// sorted_scratch[warp + 34].
	sorted_scratch[warp + 0] = fullSorted;
	sorted_scratch[warp + 1] = radixSorted;
	sorted_scratch[warp + 2] = firstValue;
	sorted_scratch[warp + 3] = lastValue;


	// Re-index the counters so the low short is bucket i and the high short
	// is bucket i + NUM_BUCKETS / 2.
	uint packed;
	if(1 == NUM_BITS) packed = counters[0];
	else {
		uint low = warpCounters[lane / 2];
		uint high = warpCounters[(lane + NUM_BUCKETS / 2) / 2];
		packed = prmt(low, high, (1 & lane) ? 0x7632 : 0x5410);
	}
	__syncthreads();

	// Gather the fullSorted and radixSorted flags for each warp.
	if(tid < NUM_WARPS) {
		sorted_scratch = sorted_shared + 65 * tid;

		uint fullSortedCount = sorted_scratch[0];
		uint radixSortedCount = sorted_scratch[1];

		int allFullSorted = __all(fullSortedCount);
		int numRadixSortedBlocks = __popc((NUM_WARPS - 1) & __ballot(fullSortedCount));

		// If these conditions are both false, then we can quit the early detection process.
		if(numRadixSortedBlocks || allFullSorted) {

			if(!tid) atomicAdd(&sortDetectCounters_global[0], numRadixSortedBlocks);

			uint firstBucket = bfe(firstValue, bit, NUM_BITS);
			
			// Zero this thread's fullSortedCount/radixSortedCount if it is not sequenced
			// with respect to the preceding one.
			if(tid) {
				uint preceding = sorted_scratch[-65 + 34];
				if(preceding > firstValue) fullSortedCount = 0;
				if(bfe(preceding, bit, NUM_BITS) > firstBucket) radixSortedCount = 0;
			}


			// Check if the entire block is sorted.
			fullSortedCount = __all(fullSortedCount);
			radixSortedCount = __all(radixSortedCount);

			if(fullSortedCount || radixSortedCount) {
				if(!tid) {
					// Load the last element of the preceding count block from global
					// memory.
					uint preceding = 0;
					if(block) preceding = keys_global[warpStart - 1].w;
					if(preceding > firstValue) fullSortedCount = 0;
					if(bfe(preceding, bit, NUM_BITS) > firstBucket) radixSortedCount = 0;

					if(fullSortedCount) atomicAdd(&sortDetectCounters_global[1], 1);
					if(radixSortedCount) atomicAdd(&sortDetectCounters_global[2], 1);
				}
			}
		}
	}

	// Set the high bit of the first packed counter in the sort block.
	if(0 == lane && radixSorted) packed |= 1<< 15;

	if(lane < NUM_CHANNELS)
		counts_shared[NUM_CHANNELS * warp + lane] = packed;
	__syncthreads();

	if(tid < NUM_WARPS * NUM_CHANNELS) {
		// write to a segment-aligned block.
		const uint WriteSpacing = RoundUp(NUM_WARPS * NUM_CHANNELS, WARP_SIZE);
		counts_global[block * WriteSpacing + tid] = counts_shared[tid];
	}
}


#undef NUM_BITS
#undef NUM_BUCKETS
#undef COUNT_FUNC
#undef SHARED_WARP_MEM
#undef COUNT_SHARED_MEM
#undef PACKED_SPACING
#undef INNER_LOOP
#undef sorted_shared


