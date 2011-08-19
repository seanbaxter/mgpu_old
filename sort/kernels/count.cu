
// Have each warp process a histogram for one block. If the block has
// NUM_VALUES, each thread process NUM_VALUES / WARP_SIZE. Eg if NUM_VALUES is
// 2048, each thread process 64 values. This operation is safe for NUM_VALUES up
// to 4096.

#define NUM_BUCKETS (1<< NUM_BITS)
#define NUM_COUNTERS ((NUM_BUCKETS + 3) / 4)
#define NUM_CHANNELS ((NUM_BUCKETS + 1) / 2)

#define PACKED_SPACING ((1 == NUM_BITS) ? 16 : 8)

// Only reserve as much shared memory as required - NUM_COUNTERS ints per
// thread. For 6 == NUM_BITS, this is 16 ints per thread, for 16384 bytes for a
// 256 thread group, allowing 50% occupancy. All smaller keys give 83% 
// occupancy, as they are register limited.

#define SHARED_WARP_MEM MAX(WARP_SIZE * NUM_COUNTERS, WARP_SIZE * 2)

#define counts_shared COUNT_SHARED_MEM

#ifdef DETECT_SORTED
// To detect data that is already sorted, reserve an additional 64 ints per 
// warp.
__shared__ volatile uint COUNT_SHARED_MEM[NUM_WARPS * (SHARED_WARP_MEM + 64)];

// sorted_shared is scratch space for testing if each segment of loaded keys are
// sorted.
#define sorted_shared (counts_shared + NUM_WARPS * SHARED_WARP_MEM)

#else
__shared__ volatile uint COUNT_SHARED_MEM[NUM_WARPS * SHARED_WARP_MEM];

#endif

#define INNER_LOOP 16

// To sorted_global, write the first key encountered and the last key 
// encountered in the count block if all all values in the count block are 
// sorted. If the values aren't all sorted, don't spend the transaction writing
// to the array.

// NOTE: for 6 == NUM_BITS, high shared memory usage allows only 33% occupancy,
// when using DETECT_SORTED. It may be possible to instead run the inner loop,
// then back up two rows of accumulate, and test for sortedness.
// More work may be required.

extern "C" __global__ 
void COUNT_FUNC(const uint* keys_global, uint bit, uint numElements, 
	uint valPerThread, uint* counts_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint warpStart = (block * NUM_WARPS + warp) * (WARP_SIZE * valPerThread);

	volatile uint* warpCounters = counts_shared + warp * SHARED_WARP_MEM;

	volatile uint* counters = warpCounters + lane;

#ifdef DETECT_SORTED
	volatile uint* sorted_scratch = sorted_shared + 64 * warp + lane;
	int fullSortedCount = 0;
	int radixSortedCount = 0;
	uint firstValue;
	uint lastValue;
#endif

	// Define the counters so we can pass them to IncBucketCounter. They don't
	// actually get used unless NUM_BITS <= 3 however.
	uint counter0 = 0;
	uint counter1 = 0;
	
	// clear all the counters
	#pragma unroll
	for(int i = 0; i < NUM_COUNTERS; ++i)
		counters[WARP_SIZE * i] = 0;

	uint values[INNER_LOOP];
	if(warpStart >= numElements) {
#ifdef DETECT_SORTED
		fullSortedCount = valPerThread;
		radixSortedCount = valPerThread;
		firstValue = 0xffffffff;
		lastValue = 0xffffffff;
#endif
	} else {

		// Unroll to read 8 values at a time
		const uint* warpData = keys_global + warpStart + lane;
		uint end = valPerThread / INNER_LOOP;

#ifdef DETECT_SORTED

		sorted_scratch[WARP_SIZE] = 0;

		for(int i = 0; i < end; ++i) {
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) 
				values[j] = warpData[j * WARP_SIZE];
			warpData += INNER_LOOP * WARP_SIZE;

			if(!i) firstValue = values[0];

			#pragma unroll
			for(int j = 0; j < INNER_LOOP; j += 2) {

				uint bucket1 = bfe(values[j], bit, NUM_BITS);

				int2 order1 = TestSortOrder(sorted_scratch, values[j], bucket1,
					bit, NUM_BITS, lane, j);

				IncBucketCounter(bucket1, counters, counter0, counter1, 
					NUM_BITS);

				uint bucket2 = bfe(values[j + 1], bit, NUM_BITS);

				int2 order2 = TestSortOrder(sorted_scratch, values[j + 1],
					bucket2, bit, NUM_BITS, lane, j + 1);

				IncBucketCounter(bucket2, counters, counter0, counter1, 
					NUM_BITS);

				// Exploit VADD to add three values together in the same 
				// instruction. Use negative assignment because SET.LE returns
				// -1 on success.
				fullSortedCount -= order1.x + order2.x;
				radixSortedCount -= order1.y + order2.y;

				// A bug in nvcc causes an explosion of register usage due to 
				// reordering of instructions to increase ILP. Incrementing
				// fullSortedCount and  radixSortedCount in the obvious way
				// pushes all the additions to the end of the dynamic loop,
				// greatly increasing register count. To prevent this, write
				// both results to shared memory every few iterations. Because
				// shared memory accesses are serialized with the volatile
				// qualifier, this forces the compiler to actually evaluate our
				// code rather than defering it. Some trials have demonstrated
				// that writing to shared mem every 3 iterations gives the best
				// performance. To avoid corrupting the sorted_scratch array 
				// with gibberish, store where the next pass will store, so
				// everything we write gets overwritten anyway.
				
				if((2 & j) && (j + 2 < INNER_LOOP))
					sorted_scratch[0] = fullSortedCount + radixSortedCount;
			}
		}
		fullSortedCount = -fullSortedCount;
		radixSortedCount = -radixSortedCount;
		lastValue = values[INNER_LOOP - 1];

#else
		// !defined(DETECT_SORTED)

		for(int i = 0; i < end; ++i) {
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) 
				values[j] = warpData[j * WARP_SIZE];
			warpData += INNER_LOOP * WARP_SIZE;

			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				uint bucket = bfe(values[j], bit, NUM_BITS);
				IncBucketCounter(bucket, counters, counter0, counter1, 
					NUM_BITS);
			}
		}
#endif

		// Write the counters to shared memory if they were stored in register.
	#if NUM_BITS <= 2
		counters[0] = counter0;
	#elif 3 == NUM_BITS
		counters[0] = counter0;
		counters[WARP_SIZE] = counter1;
	#endif
	}

	// If there are multiple counters, run GatherSums until we only have one
	// row of counters remaining.
#if 1 == NUM_BITS
	// 16-bit packing was used from the start, so we go directly to parallel 
	// reduction.
#elif 2 == NUM_BITS
	// Grab even and odd counters, and add and widen
	uint a = warpCounters[~1 & lane];
	uint b = warpCounters[1 | lane];
	uint gather = (1 & lane) ? 0x4342 : 0x4140;
	uint sum = prmt(a, 0, gather) + prmt(b, 0, gather);
	counters[0] = sum;
#elif 3 == NUM_BITS
	// At this point we have to manually unroll the GatherSums calls, because
	// nvcc is stupid and complains "Advisory: Loop was not unrolled, not an
	// innermost loop." This is due to the branch logic in GatherSums.
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

	// There are probably still multiple copies of each sum. Perform a parallel
	// scan to add them all up.
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

#ifdef DETECT_SORTED

	// Use ballot to check if each sort block is internally sorted.
	int fullSorted = __all(valPerThread == fullSortedCount);
	int radixSorted = __all(valPerThread == radixSortedCount);

	// Counters from each sort block will be accessed by different threads
	// in the first warp. To eliminate bank conflicts, space the values out
	// by adding the warp counter to sorted_scratch for each thread.

	// The last value of the last thread in the warp is at
	// sorted_scratch[warp + 2 + WARP_SIZE - 1], i.e.
	// sorted_scratch[warp + 34].
	sorted_scratch[warp + 0] = fullSorted;
	sorted_scratch[warp + 1] = radixSorted;
	sorted_scratch[warp + 2] = firstValue;
	sorted_scratch[warp + 3] = lastValue;

#endif
	
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

#ifdef DETECT_SORTED

	// Gather the fullSorted and radixSorted flags for each warp.
	if(tid < NUM_WARPS) {
		sorted_scratch = sorted_shared + 65 * tid;

		uint fullSortedCount = sorted_scratch[0];
		uint radixSortedCount = sorted_scratch[1];

		int allFullSorted = __all(fullSortedCount);
		int numRadixSortedBlocks = __popc(((1<< NUM_WARPS) - 1) & 
			__ballot(radixSortedCount));

		// If these conditions are both false, then we can quit the early
		// detection process.
		if(numRadixSortedBlocks || allFullSorted) {
			if(!tid) 
				atomicAdd(&sortDetectCounters_global[0], numRadixSortedBlocks);

			firstValue = sorted_scratch[2];
			uint firstBucket = bfe(firstValue, bit, NUM_BITS);
			
			// Zero this thread's fullSortedCount/radixSortedCount if it is not
			// sequenced with respect to the preceding one.
			if(tid) {
				uint preceding = sorted_scratch[-65 + 34];
				if(preceding > firstValue) 
					fullSortedCount = 0;
				if(bfe(preceding, bit, NUM_BITS) > firstBucket) 
					radixSortedCount = 0;
			}


			// Check if the entire block is sorted.
			fullSortedCount = __all(fullSortedCount);
			radixSortedCount = __all(radixSortedCount);

			if(fullSortedCount || radixSortedCount) {
				if(!tid) {
					// Load the last element of the preceding count block from 
					// global memory.
					uint preceding = 0;
					if(block)
						preceding = keys_global[warpStart - 1];
					if(preceding > firstValue) 
						fullSortedCount = 0;
					if(bfe(preceding, bit, NUM_BITS) > firstBucket) 
						radixSortedCount = 0;

					if(fullSortedCount) 
						atomicAdd(&sortDetectCounters_global[1], 1);
					if(radixSortedCount) 
						atomicAdd(&sortDetectCounters_global[2], 1);
				}
			}
		}
	}

	// Set the high bit of the first packed counter in the sort block.
	if(!lane && radixSorted) packed |= 1<< 15;

#endif

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
