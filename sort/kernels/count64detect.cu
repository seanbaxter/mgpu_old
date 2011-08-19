// The count kernel has problems with NUM_BITS = 6 and DETECT_SORTED defined.
// Without sort detection, the kernel uses 1 byte of shared mem per thread per
// bucket. With 64 buckets (6 bits), that's 64 bytes per thread. 48k / 64 bytes
// allows an occupancy of 768 threads, or 50%. This is sufficient for the kernel
// to perform well. However the extra 64 slots per warp (8 bytes per thread)
// brings per-block usage to 18432 bytes. This means only two blocks fit in the 48k
// window, for 33% occupancy. Because this is a kernel with low thread-level 
// parallelism (read a value from shared memory, modify it, write it back, repeat),
// the SM idles.

// This specialized kernel backs up two values of shared memory to register at
// the end of each inner loop, performs the sort detection in the same shared
// memory space, then restores the counts and continues the outer loop.

#define NUM_BUCKETS 64
#define NUM_COUNTERS 16
#define NUM_CHANNELS 32

// Only reserve as much shared memory as required - NUM_COUNTERS ints per thread.
// For 6 == NUM_BITS, this is 16 ints per thread, for 16384 bytes for a 256 thread
// group, allowing 50% occupancy. All smaller keys give 83% occupancy, as they
// are register limited.

// By not going over this count we achieve 50% occupancy.
 #define counts_shared COUNT_SHARED_MEM

// To detect data that is already sorted, reserve an additional 64 ints per warp.
__shared__ volatile uint COUNT_SHARED_MEM[NUM_THREADS * NUM_COUNTERS];


extern "C" __global__ void COUNT_FUNC(const uint* keys_global, uint bitOffset, 
	uint numElements, uint valuesPerThread, uint* counts_global, uint2* sortDetect_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint warpStart = (block * NUM_WARPS + warp) * (WARP_SIZE * valuesPerThread);

	volatile uint* warpCounters = counts_shared + warp * SHARED_WARP_MEM;
	volatile uint* counters = warpCounters + lane;

	// Space the scratch space 65 slots out to avoid conflict-free access to the first
	// and last sort block values.
	volatile uint* sorted_scratch = sorted_shared + 64 * warp + lane;
	int threadSortedCount = 0;

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

#ifdef DETECT_SORTED
	uint firstSortedBucket;
#endif

	uint buckets[INNER_LOOP];
	if(warpStart >= numElements) {
		// Set the last bucket with all the values.
		counters[(NUM_COUNTERS - 1) * WARP_SIZE] = valuesPerThread<< (32 - PACKED_SPACING);
#ifdef DETECT_SORTED
		threadSortedCount = valuesPerThread;
		firstSortedBucket = NUM_BUCKETS - 1;
		buckets[INNER_LOOP - 1] = NUM_BUCKETS - 1;
#endif
	} else {

		// Unroll to read 8 values at a time
		const uint* warpData = keys_global + warpStart + lane;

#ifdef DETECT_SORTED
		// Clear the sorted_scratch space to avoid false failure on the first values.
		sorted_scratch[WARP_SIZE] = 0;
#endif

		uint end = valuesPerThread / INNER_LOOP;
		for(int i = 0; i < end; ++i) {
		
			// Extract the bits to be sorted and keep in a register array.
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				// extract the current bucket
				uint key = warpData[j * WARP_SIZE];
				buckets[j] = bfe(key, bitOffset, NUM_BITS);
			}
			warpData += INNER_LOOP * WARP_SIZE;

			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				// Increment the actual bucket count for the histogram.
				IncBucketCounter(buckets[j], counters, counter0, counter1, NUM_BITS);
		
#if 5 == NUM_BITS && defined(DETECT_SORTED)
				threadSortedCount += TestBucketOrder(sorted_scratch, buckets[j], lane, j);
#endif
			}
			

#ifdef DETECT_SORTED
			if(!i) firstSortedBucket = buckets[0];

	#if NUM_BITS < 5
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				threadSortedCount += TestBucketOrder(sorted_scratch, buckets[j], lane, j);
			}
	#endif
#endif
		}



		/*
		// Cascade the counter loop by loading an INNER_LOOP ahead of where we 
		// are processing.
		uint keys[INNER_LOOP];
		for(int j = 0; j < INNER_LOOP; ++j)
			keys[j] = warpData[j * WARP_SIZE];
		warpData += INNER_LOOP * WARP_SIZE;

		uint end = valuesPerThread / INNER_LOOP - 1;
		for(int i = 0; i < end; ++i) {
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				buckets[j] = bfe(keys[j], bitOffset, NUM_BITS);
				keys[j] = warpData[j * WARP_SIZE];
			}

#ifdef DETECT_SORTED
			if(!i) firstSortedBucket = buckets[0];
#endif
			warpData += INNER_LOOP * WARP_SIZE;

			// Process buckets while loading keys.
			#pragma unroll
			for(int j = 0; j < INNER_LOOP; ++j) {
				IncBucketCounter(buckets[j], counters, counter0, counter1, NUM_BITS);
#ifdef DETECT_SORTED
				threadSortedCount += TestBucketOrder(sorted_scratch, buckets[j], lane, j);
#endif
			}
		}

		// Process the last set of keys
		#pragma unroll
		for(int j = 0; j < INNER_LOOP; ++j) {
			buckets[j] = bfe(keys[j], bitOffset, NUM_BITS);
			IncBucketCounter(buckets[j], counters, counter0, counter1, NUM_BITS);
#ifdef DETECT_SORTED
			threadSortedCount += TestBucketOrder(sorted_scratch, buckets[j], lane, j);
#endif
		}
*/
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

#ifdef DETECT_SORTED
	// Use ballot to check if each sort block is internally sorted.
	int threadSorted = valuesPerThread == threadSortedCount;
	int isSorted = __all(threadSorted);

	// Counters from each sort block will be accessed by different threads
	// in the first warp. To eliminate bank conflicts, space the values out
	// by adding the warp counter to sorted_scratch for each thread.

	// Store the isSorted flag in slot 0.
	sorted_scratch[warp + 0] = isSorted;

	// Store the first value encountered in slot 1.
	sorted_scratch[warp + 1] = firstSortedBucket;

	// Store the last value encountered.
	sorted_scratch[warp + 2] = buckets[INNER_LOOP - 1];
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

	if(tid < WARP_SIZE) {
		uint sortBlockSorted = 0;
		uint countBlockSorted = 1;

		if(tid < NUM_WARPS) {
			sorted_scratch = sorted_shared + 65 * tid;
			sortBlockSorted = sorted_scratch[0];
			countBlockSorted = sortBlockSorted;
			
			// Does this sort block begin after the end of the preceding one?
			if(tid) {
				uint sortBlockFirst = sorted_scratch[1];
				uint sortBlockPrevLast = sorted_scratch[-65 + 2 + 31];
				if(sortBlockPrevLast > sortBlockFirst) countBlockSorted = 0;
			} 
		}

		sortBlockSorted = __popc(__ballot(sortBlockSorted));
		countBlockSorted = __all(countBlockSorted);
		if(0 == tid) {
			if(sortBlockSorted) 
				atomicAdd(&sortDetectCounters_global[0], sortBlockSorted);
			if(countBlockSorted) {
				atomicAdd(&sortDetectCounters_global[1], 1);

				// Output the first and last values of the block to global memory.
				uint sortBlockFirst = sorted_shared[1];
				uint sortBlockLast = sorted_shared[65 * (NUM_WARPS - 1) + WARP_SIZE + 1];

				// To save a transaction, issue a 8-byte store.
				sortDetect_global[block] = make_uint2(sortBlockFirst, sortBlockLast);
			}
		}
	}

	// Set the high bit of the first packed counter in the sort block.
	if(0 == lane && isSorted) packed |= 1<< 15;

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
