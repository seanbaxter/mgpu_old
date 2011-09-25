
////////////////////////////////////////////////////////////////////////////////
// INTER-WARP REDUCTION 
// Calculate the length of the last segment in the last lane in each warp. Also
// store the block offset to shared memory for the next pass.

template<int NumWarps>
DEVICE2 uint BlockScan(uint tid, uint warp, uint lane, uint last,
	uint warpFlags, uint mask, volatile uint* blockOffset_shared) {

	const int LogNumWarps = LOG_BASE_2(NumWarps);

	__shared__ volatile uint blockShared[3 * NumWarps];
	if(WARP_SIZE - 1 == lane) {
		blockShared[NumWarps + warp] = last;
		blockShared[2 * NumWarps + warp] = warpFlags;
	}
	__syncthreads();

	if(tid < NumWarps) {
		// Pull out the sum and flags for each warp.
		volatile uint* s = blockShared + NumWarps + tid;
		uint warpLast = s[0];
		uint flag = s[NumWarps];
		s[-NumWarps] = 0;

		uint blockFlags = __ballot(flag);

		// Mask out the bits at or above the current warp.
		blockFlags &= mask;

		// Find the distance from the current warp to the warp at the start of 
		// this segment.
		int preceding = 31 - __clz(blockFlags);
		uint distance = tid - preceding;


		// INTER-WARP reduction
		uint warpSum = warpLast;
		uint warpFirst = blockShared[NumWarps + preceding];

		#pragma unroll
		for(int i = 0; i < LogNumWarps; ++i) {
			uint offset = 1<< i;
			if(distance > offset) warpSum += s[-offset];
			if(i < LogNumWarps - 1) s[0] = warpSum;
		}
		// Subtract warpLast to make exclusive and add first to grab the
		// fragment sum of the preceding warp.
		warpSum += warpFirst - warpLast;

		// Store warpSum back into shared memory. This is added to all the
		// lane sums and those are added into all the threads in the first 
		// segment of each lane.
		blockShared[tid] = warpSum;

		// Set the block offset for the next brick of data.
		if(NumWarps - 1 == tid) {
			if(!flag) warpLast += warpSum;
			*blockOffset_shared = warpLast;
		}
	}
	__syncthreads();

	return blockShared[warp];
}


////////////////////////////////////////////////////////////////////////////////
// Segmented scan downsweep logic. Abstracts away loading of values and head 
// flags.

template<int NumWarps, int ValuesPerThread>
DEVICE2 void SegScanDownsweep(uint tid, uint lane, uint warp, 
	uint x[ValuesPerThread], const uint flags[ValuesPerThread],
	volatile uint* warpShared, volatile uint* threadShared, bool inclusive, 
	volatile uint* blockOffset_shared) {

	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP PASS
	// Add sum to all the values in the continuing segment (that is, before the
	// first start flag) in this thread.

	uint blockOffset = 0;
	if(!tid) blockOffset = *blockOffset_shared;
	uint last = blockOffset;

	// Compute the exclusive scan into scan. These values are then added to the
	// final thread offsets after the inter-warp multiscan pattern.
	uint hasHeadFlag = 0;

	if(inclusive) {
		#pragma unroll
		for(int i = 0; i < ValuesPerThread; ++i) {
			if(flags[i]) last = 0;
			hasHeadFlag |= flags[i];

			x[i] += last;
			last = x[i];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < ValuesPerThread; ++i) {
			if(flags[i]) last = 0;
			if(flags[i]) hasHeadFlag |= flags[i];
			uint val = x[i];

			x[i] = last;
			last += val;
		}
	}

	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP SEGMENT PASS
	// Run a ballot and clz to find the lane containing the start value for the
	// segment that begins this thread.

	uint warpFlags = __ballot(hasHeadFlag);

	// Mask out the bits at or above the current thread.
	uint mask = bfi(0, 0xffffffff, 0, lane);
	uint warpFlagsMask = warpFlags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	int preceding = 31 - __clz(warpFlagsMask);
	uint distance = lane - preceding;


	////////////////////////////////////////////////////////////////////////////
	// REDUCTION PASS
	// Run a prefix sum scan over last to compute for each lane the sum of all
	// values in the segmented preceding the current lane, up to that point.
	// This is added back into the thread-local exclusive scan for the continued
	// segment in each thread.

	volatile uint* shifted = threadShared + 1;
	shifted[-1] = 0;
	shifted[0] = last;
	uint sum = last;
	uint first = warpShared[1 + preceding];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(distance > offset) sum += shifted[-offset];
		if(i < LOG_WARP_SIZE - 1) shifted[0] = sum;
	}
	// Subtract last to make exclusive and add first to grab the fragment
	// sum of the preceding thread.
	sum += first - last;

	// Call BlockScan for inter-warp scan on the reductions of the last
	// segment in each warp.
	uint lastSegLength = last;
	if(!hasHeadFlag) lastSegLength += sum;

	uint blockScan = BlockScan<NumWarps>(tid, warp, lane, lastSegLength,
		warpFlags, mask, blockOffset_shared);
	if(!warpFlagsMask) sum += blockScan;

	#pragma unroll
	for(int i = 0; i < ValuesPerThread; ++i) {
		if(flags[i]) sum = 0;
		x[i] += sum;
	}
}
