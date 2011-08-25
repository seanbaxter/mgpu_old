////////////////////////////////////////////////////////////////////////////////
// PHASE ONE
// Build a running sum of all the bucket counts per column. This is a vertical 
// sum. Then perform a horizontal add from left-to-right to get the total number
// of buckets for this histogram block.

// Input:
// bucketCount_global is the output of the count kernel.
//		Bucket counts for each sort block packed into 16bit uints.
// rangePairs_global is the start and end interval for each warp in this pass.
//		These values always refers to warp's worth of data, so multiply by 32.
// Output:
// countScan_global is the total count for each bucket. Output for each
//		histogram block is NUM_BUCKETS, totalling NumSMs * NUM_BUCKETS. These 
//		values are scanned and modified in-place by the phase 2 histogram 
//		kernel.
// columnScan_global is the scan of bucket counts for totals within each warp of
//		this block. This is required as it gives each warp in the phase 3 
//		histogram kernel a starting scatter offset for each bucket. This data
//		is not read by the phase 2 kernel. However, the countScan_globals are 
//		modified by the phase 2 kernel and added to columnScan_global in phase
//		3.
//		
// The 16-bit totals from the count kernel are expanded to 32 bits in this
// kernel.


extern "C" __global__ void HISTOGRAM_FUNC1(const uint* bucketCount_global,
	const uint2* rangePairs_global, uint* countScan_global,
	uint* columnScan_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	
	uint2 range = rangePairs_global[NUM_WARPS * block + warp];

	// running sum only for this thread
	uint countLow = 0;
	uint countHigh = 0;

	uint current = range.x;
	while(current < range.y) {
		uint packed = bucketCount_global[WARP_SIZE * current + lane];

		// The most sig bit of the lo short is a flag indicating sort detect.
		// Clear this bit in hist1, and read it in hist3.
		countLow += 0x00007fff & packed;
		countHigh = shr_add(packed, 16, countHigh);
		++current;
	}

	// Store the low counts at tid and the high counts at NUM_THREADS + tid.
	hist_shared1[tid] = countLow;
	hist_shared1[NUM_THREADS + tid] = countHigh;

#if NUM_SORT_BLOCKS_PER_WARP > 1
	if(lane < NUM_CHANNELS) {
		for(int i = 1; i < NUM_SORT_BLOCKS_PER_WARP; ++i) {
			countLow += hist_shared1[tid + i * NUM_CHANNELS];
			countHigh += hist_shared1[NUM_THREADS + tid + i * NUM_CHANNELS];
		}
		hist_shared1[tid] = countLow;
		hist_shared1[NUM_THREADS + tid] = countHigh;
	}
#endif
	__syncthreads();

	// Perform inter-warp scan for bucket totals for each warp.
	if(tid < NUM_BUCKETS) {
		volatile uint* shared = hist_shared1 + ((NUM_CHANNELS - 1) & tid);
		if(tid >= NUM_CHANNELS) shared += NUM_THREADS;

		uint index = NUM_BUCKETS * NUM_WARPS * block + 
			((NUM_CHANNELS - 1) & tid);
		if(tid >= NUM_CHANNELS) index += NUM_CHANNELS * NUM_WARPS;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < NUM_WARPS; ++i) {
			// Output the scan values (not perfectly coalesced)
			columnScan_global[index + i * NUM_CHANNELS] = x;
			uint y = shared[i * WARP_SIZE];
			x += y;
		}

		// Output the bucket totals
		countScan_global[block * NUM_BUCKETS + tid] = x;
	}
}

