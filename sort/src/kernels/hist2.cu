
// Load in countScan_global from all 

extern "C" __global__ void HISTOGRAM_FUNC2(uint numBlocks, 
	uint* countScan_global) {

	// Perform an in-place scan over countScan_global.
	uint tid = threadIdx.x;
	int numCounts = NUM_BUCKETS * numBlocks;

	for(int i = tid; i < numCounts; i += NUM_THREADS)
		blockScan_shared2[i] = countScan_global[i];
	__syncthreads();

	// Run a sequential scan over all the blocks. This isn't optimal, but it's 
	// a trivial amount of data to process and we should keep this simple.
	if(tid < NUM_BUCKETS) {	
		uint x = 0;
		for(int i = tid; i < numCounts; i += NUM_BUCKETS) {
			uint y = blockScan_shared2[i];
			blockScan_shared2[i] = x;
			x += y;
		}

		// Write the total number of values in each bucket. The host code can 
		// cuMemcpy this array and figure out if the array is already fully or 
		// partially sorted.
		bucketTotals_shared2[tid] = x;
		// bucketTotals_global[tid] = x;
	}
	__syncthreads();

	
	// Run a parallel scan to sum up all values of x
#if NUM_BITS <= 5
	if(tid < NUM_BUCKETS) {
		volatile uint* totals = bucketTotals_shared2 + tid;
		uint sum = totals[0];
		uint x = sum;
		#pragma unroll
		for(int i = 0; i < NUM_BITS; ++i) {
			uint offset = 1<< i;
			uint y = totals[-offset];
			if(tid >= offset) x += y;
			if(i < NUM_BITS - 1) totals[0] = x;
		}
		totals[0] = x - sum;
	}
#elif 6 == NUM_BITS
	if(tid < WARP_SIZE) {
		volatile uint* totals = bucketTotals_shared2 + tid;
		uint sum1 = totals[0];
		uint sum2 = totals[WARP_SIZE];
		uint x1 = sum1;
		uint x2 = sum2;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			uint offset = 1<< i;
			uint y1 = totals[-offset];
			uint y2 = totals[WARP_SIZE - offset];
			if(tid >= offset) x1 += y1;
			x2 += y2;			
			if(i < LOG_WARP_SIZE - 1) totals[0] = x1, totals[WARP_SIZE] = x2;
		}
		x2 += x1;
		totals[0] = x1 - sum1;
		totals[WARP_SIZE] = x2 - sum2;
	}

	if(tid < 64)
		countScan_global[tid] = bucketTotals_shared2[tid];
#endif
	__syncthreads();

	// Add the exclusive bucket sums to the block offsets and write back to
	// countScan_global.
	for(int i = tid; i < numCounts; i += NUM_THREADS) {
		uint bucketOffset = bucketTotals_shared2[(NUM_BUCKETS - 1) & i];
		countScan_global[i] = blockScan_shared2[i] + bucketOffset;
	}
}



////////////////////////////////////////////////////////////////////////////////
// PHASE TWO
// After all blocks synchronize after writing their total bucket counts, the
// first block performs an in-place inclusive scan of the bucket counts. This 
// is a straight-forward linear scan.

// Lots of __syncthreads but who cares? On all current hardware (16 SMs or
// fewer), it only takes one inner loop (64 buckets * 16 SM = 1024, the max 
// number of threads).
	

#if 0

extern "C" __global__ void HISTOGRAM_FUNC2(uint end, uint* countScan_global,
	uint* bucketScan_global) {
			
	// load in NUM_THREADS values from countScan_global, perform a parallel 
	// scan, add in the bucket totals from the preceding scans, and write-back 
	// to countScan_global
	uint tid = threadIdx.x;
	uint current = 0;
	// uint end = NUM_BUCKETS * gridDim.x;
	uint blockTotal = 0;
	
	while(current < end) {
		uint index = current + tid;
		uint load = min(index, end - 1);
		uint count = countScan_global[load];
		if(index > load) count = 0;
		
		// put the count in shared memory
		blockTotals_shared2[tid] = count;
		uint x = count;
		__syncthreads();
		
		// scan to add the counts up			
		#pragma unroll
		for(uint i = 0; i < REDUCTION_ITER - 1; ++i) {
			uint offset = NUM_BUCKETS<< i;
			uint left = (NUM_THREADS - 1) & (tid - offset);
			uint y = blockTotals_shared2[left];
			if(offset <= tid) x += y;
			__syncthreads();
			blockTotals_shared2[tid] = x;
			__syncthreads();
		}
		
		// subtract count from running sum to get exclusive sum
		// add the preceding totals and write back to histogram_global
		if(index == load) countScan_global[index] = x - count + blockTotal;

		uint i = NUM_THREADS - NUM_BUCKETS + ((NUM_BUCKETS - 1) & tid);
		blockTotal += blockTotals_shared2[i];
		current += NUM_THREADS;
		__syncthreads();
	}
			
	
	// This section is macro hell because nvcc is really stupid. It complains 
	// about "pointless comparison using unsigned integer with zero" and
	// complains.

	// Scan the block totals
#if NUM_BUCKETS <= WARP_SIZE
	if(tid < NUM_BUCKETS) {
		volatile uint* scan = blockTotals_shared2 + NUM_THREADS - NUM_BUCKETS;
		uint count = scan[tid];
		uint x = count;

		#pragma unroll
		for(int i = 0; i < NUM_BITS; ++i) {
			uint offset = 1<< i;
			uint left = (WARP_SIZE - 1) & (tid - offset);
			uint y = scan[left];
			if(offset <= tid) x += y;
			scan[tid] = x;
		}

		// write exclusive bucket sum
		bucketScan_global[tid] = x - count;
	}
#elif NUM_BUCKETS == 2 * WARP_SIZE

	// NUM_BUCKETS > WARP_SIZE
	// Each lane manages two bucket counts
	if(tid < WARP_SIZE) {
		volatile uint* scan = blockTotals_shared2 + NUM_THREADS - NUM_BUCKETS;
		uint count1 = scan[tid];
		uint count2 = scan[WARP_SIZE + tid];
		uint x1 = count1;
		uint x2 = count2;

		#pragma unroll
		for(uint i = 0; i < NUM_BITS - 1; ++i) {
			uint offset = 1<< i;
			uint left1 = (NUM_BUCKETS - 1) & (tid - offset);
			uint left2 = (NUM_BUCKETS - 1) & (tid + WARP_SIZE - offset);
			uint y1 = scan[left1];
			uint y2 = scan[left2];
			if(offset <= tid) x1 += y1;
			if(offset <= tid + WARP_SIZE) x2 += y2;
			scan[tid] = x1;
			scan[WARP_SIZE + tid] = x2;
		}
		// add low into high
		x2 += x1;
		bucketScan_global[tid] = x1 - count1;
		bucketScan_global[WARP_SIZE + tid] = x2 - count2;
	}
#endif

}

#endif // 0
