
////////////////////////////////////////////////////////////////////////////////
// HISTOGRAM PASS

DEVICE uint MapToScan(uint k, uint x1, uint count1, uint x2, uint count2) {
	// Find the bucket that k maps to. This is the last bucket that starts at
	// or before k. This should handle empty buckets with an offset starting
	// at k.
	uint low = __ballot(k >= x1);
	uint high = __ballot(k >= x2);

	uint kLow = __clz(low);
	uint kHigh = __clz(high);
	uint bucket = high ? (63 - kHigh) : (31 - kLow);

	return bucket;
}

//DEVICE void SelectIntervalScan(const uint* counts_global,
//	uint* scanWarp_global, uint k1, uint k2, uint lane, uint warp, 
//	volatile uint* shared,

// Run a scan on only the counts in bucket b from a single warp. May be called
// 
DEVICE void SelectScanWarp(const uint* counts_global, uint* scanWarp_global,
	uint b, uint lane, uint numWarps, volatile uint* shared) {

	shared[lane] = 0;
	volatile uint* s = shared + lane + WARP_SIZE / 2;
	uint offset = 0;

	for(int i = 0; i < numWarps; i += WARP_SIZE) {
		uint source = i + lane;
		uint count = 0;
		if(source < numWarps)
			count = counts_global[64 * source + b];

		// Parallel scan and write back to global memory.
		uint x = count;		
		s[0] = x;

		#pragma unroll
		for(int j = 0; j < LOG_WARP_SIZE; ++j) {
			int offset = 1<< j;
			x += s[-offset];
			s[0] = x;
		}

		// Make a coalesced write to scanWarp_global.
		scanWarp_global[i + lane] = offset + x - count;
		offset += shared[47];
	}
}

DEVICE void SelectHist(const uint* counts_global, uint* scanTotal_global,
	int numWarps, uint k1, uint k2, bool interval, uint* bucket1Scan_global, 
	uint* bucket2Scan_global, uint* intervalScan_global) {

	const int NumThreads = 1024;
	const int NumWarps = NumThreads / WARP_SIZE;

	__shared__ volatile uint shared[4 * 1024];
	volatile uint* counts_shared = shared;
	volatile uint* scan_shared = shared + 2 * 1024;
	volatile uint* b1_shared = scan_shared + 64;
	volatile uint* b2_shared = b1_shared + 1;

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	////////////////////////////////////////////////////////////////////////////
	// Loop over all counters and accumulate counts for each radix digits. This
	// is the sequential part.
	// Each thread maintains two counters (low and high lanes of the count).
	uint totals[2] = { 0, 0 };

	int curWarp = warp;
	while(curWarp < numWarps) {
		uint x1 = counts_global[64 * curWarp + lane];
		uint x2 = counts_global[64 * curWarp + 32 + lane];

		totals[0] += x1;
		totals[1] += x2;

		curWarp += NumWarps;
	}
	counts_shared[64 * warp + lane] = totals[0];
	counts_shared[64 * warp + 32 + lane] = totals[1];
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Have the first 64 threads run through shared memory and get final counts.
	// This is the parallel part.

	if(tid < 64) {
		uint total = counts_shared[tid];

		#pragma unroll
		for(int i = 64; i < 2048; i += 64)
			total += counts_shared[tid + i];

		scan_shared[tid] = total;

		// Write the totals to global memory.
		scanTotal_global[tid] = total;
	}
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Parallel scan the radix digit totals. This is needed to find which radix
	// digit to expect k1 and k2 in. Those are stored in b1_shared and
	// b2_shared.

	// Scan the totals.
	if(tid < WARP_SIZE) {
		uint sum1 = scan_shared[tid];
		uint sum2 = scan_shared[WARP_SIZE + tid];

		uint x1 = sum1;
		uint x2 = sum2;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			
			if(tid >= offset) x1 += scan_shared[tid - offset];
			x2 += scan_shared[WARP_SIZE + tid - offset];
			
			scan_shared[tid] = x1;
			scan_shared[WARP_SIZE + tid] = x2;
		}
		x2 += scan_shared[tid];

		x1 -= sum1;
		x2 -= sum2;

		// Write the scan to global memory.
		scanTotal_global[64 + tid] = x1;
		scanTotal_global[64 + WARP_SIZE + tid] = x2;

		uint b1 = MapToScan(k1, x1, sum1, x2, sum2);
		if(!tid) {
			*b1_shared = b1;
			scanTotal_global[2 * 64 + 0] = b1;
		}

		if(interval) {
			uint b2 = MapToScan(k2, x1, sum1, x2, sum2);
			if(!tid) {
				*b2_shared = b2;
				scanTotal_global[2 * 64 + 1] = b2;
			}
		}
	}
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Generate the warp scan offsets. There are three cases for this:
	// 1) Both k1 and k2 map to the same bucket. Only generate a single warp
	// scan array.
	// 2) k1 and k2 map to adjacent buckets. Here we scan only buckets b1 and 
	// b2.
	// 3) k1 and k2 map to adjacent buckets. Here we scan buckets b1, b2, and
	// also the sum of counts inside the interval over all warps.

	uint b1 = *b1_shared;
	uint b2;
	if(interval) b2 = *b2_shared;

	if(!interval || (b1 == b2)) {
		if(tid < 32) 
			SelectScanWarp(counts_global, bucket1Scan_global, b1, lane,
				numWarps, shared);		
	} else if(b1 + 1 == b2) {
		if(tid < 64) {
			uint b = warp ? b2 : b1;
			volatile uint* s = warp ? (shared + 128) : shared;
			uint* scanGlobal = warp ? bucket1Scan_global : bucket2Scan_global;

			// Run the k-smallest scan on the first two warps in parallel.
			SelectScanWarp(counts_global, scanGlobal, b, lane, numWarps, s);
		}
	} else {



	}

}



extern "C" __global__ __launch_bounds__(1024, 1)
void SelectHistValue(const uint* counts_global, uint* scanTotal_global,
	uint* scanWarps_global, int numWarps, uint k) {

	SelectHist(counts_global, scanTotal_global, numWarps, k, 0, false,
		scanWarps_global, 0, 0);
}

extern "C" __global__ __launch_bounds__(1024, 1)
void SelectHistInterval(const uint* counts_global, uint* scanTotal_global,
	uint* scanWarps_global, int numWarps, uint k1, uint k2) {

	SelectHist(counts_global, scanTotal_global, numWarps, k1, k2, true,
		scanWarps_global, scanWarps_global + numWarps, 
		scanWarps_global + 2 * numWarps);
}

