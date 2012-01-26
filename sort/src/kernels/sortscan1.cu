#pragma once

DEVICE void SortScatter1(uint tid, Values digits, uint numThreads,
	uint scatter[4], uint* scratch_shared, uint* debug_global) {

	const int NumValues = VALUES_PER_THREAD * numThreads;
	const int NumWarps = numThreads / WARP_SIZE;

	// Allocate 1 int for each thread to store its digit count. These are
	// strided for fast parallel scan access, so consume 33 values per warp.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// In the tid < WARP_SIZE part, do a sequential scan with NumWarps elements
	// per thread.
	const int StreamLen = NumWarps;
	const int StreamsPerWarp = WARP_SIZE / NumWarps;
	
	// Store the stream totals and do a parallel scan. Allocate a warp and a
	// half for the parallel scan.
	// const int ParallelScanSize = WARP_SIZE + 16;
	volatile uint* parallelScan_shared = predInc_shared + ScanSize + 16;


	uint warp = tid / WARP_SIZE;

	// Compute the number of set bits.
	uint predInc = 0;
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		predInc += digits[v];

	// Reserve space for the scan, with each warp distanced out 33 elements to 
	// avoid bank conflicts.
	volatile uint* scan = predInc_shared + tid + warp;

	scan[0] = predInc;
	__syncthreads();

	// Perform sequential scan over the bit counts.
	// The sequential operation exhibits very little ILP (only the addition and
	// next LDS can be run in parallel). This is the major pipeline bottleneck
	// for the kernel. We need to launch enough blocks to hide this latency.
	if(tid < WARP_SIZE) {
		// Each stream begins on a different bank using this indexing.
		volatile uint* scan2 = predInc_shared + StreamLen * tid + 
			tid / StreamsPerWarp;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// Write the end-of-stream total, then perform a parallel scan.
		volatile uint* scan3 = parallelScan_shared + tid;	
		scan3[-16] = 0;
		uint sum = x;
		scan3[0] = x;

		// Perform a single parallel scan over one warp of data.
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			uint y = scan3[-offset];
			
			// No conditional required because the 16 values preceding the start
			// of the scan array were zero'd out, so when tid < offset, we just
			// add in a harmless 0.
			x += y;
			
			scan3[0] = x;
		}

		// Adjust x to get the exclusive scan of set bits.
		x -= sum;

		// Store the total number of encountered set bits in the high short of
		// x.
		x = bfi(x, parallelScan_shared[WARP_SIZE - 1], 16, 16);

		// Put this packed counter back into memory to communicate to all the
		// threads in the block.
		parallelScan_shared[tid] = x;
	}
	__syncthreads();

	// Update predInc to be an exclusive scan of all the encountered set bits up
	// to this thread in the multi-scan list.
	predInc = scan[0];

	// Retrieve the packed integer containing the total number of encountered
	// set bits, and the exclusive scan of encountered set bits up to the
	// current list.
	uint packed = parallelScan_shared[tid / StreamLen];
	uint setExc = 0x0000ffff & packed;
	uint setTotal = packed>> 16;

	// Add into setExc the exclusive scan of all the lists up to this thread.
	setExc += predInc;

	// Before this thread, there are setExc encountered set bits. This implies
	// that all the other values in the preceding threads are cleared bits. This
	// count is the offset for the first cleared bit key in this thread.
	uint exc0 = VALUES_PER_THREAD * tid - setExc;

	// The offset for the first set bit key in this thread comes after all the
	// cleared bit keys in the entire block (NUM_VALUES - setTotal) plus the
	// number of preceding set bits.
	uint exc1 = NumValues - setTotal + setExc;

	// Pack exc0 into the low short and exc1 into the high short.
	uint next = bfi(exc0, exc1, 16, 16);

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD / 2; ++v) {
		uint b = 16 * digits[2 * v];
		uint offset1 = bfe(next, b, 16);
		next = shl_add(1, b, next);

		b = 16 * digits[2 * v + 1];
		uint offset2 = bfe(next, b, 16);
		next = shl_add(1, b, next);

		// Pack offset1 and offset2 together and store in scatter.
		scatter[v] = bfi(offset1, offset2, 16, 16);
	}
}

