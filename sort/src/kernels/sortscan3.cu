#pragma once

DEVICE uint4 MultiScan3(uint tid, uint2 predInc, uint numThreads, 
	uint2 bucketsPacked, uint2& offsetsPacked, uint* scratch_shared,
	uint* debug_global) {

	const int NumWarps = numThreads / WARP_SIZE;

	// Store two values per thread. These are 8 counters packed into 2 ints.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = NumWarps;
	const int StreamsPerWarp = WARP_SIZE / NumWarps;

	// Store the stream totals and do a parallel scan. We need to scan 128
	// elements, as each stream total now takes a full 16 bits.
	// const ParallelScanSize = 4 * WARP_SIZE + 32;
	volatile uint* parallelScan_shared = predInc_shared + 2 * ScanSize + 16;

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	// The first half-warp sums counts 0-3, the second half-warp sums counts
	// 4-7.
	volatile uint* scan = predInc_shared + tid + warp;
	scan[0] = predInc.x;
	scan[ScanSize] = predInc.y;
	__syncthreads();

	// Perform sequential scan over the byte-packed counts.
	// The sequential operation exhibits very little ILP (only the addition and
	// next LDS can be run in parallel). This is the major pipeline bottleneck
	// for the kernel. We need to launch enough blocks to hide this latency.

	uint x0, x1;
	if(tid < 2 * WARP_SIZE) {
		// Counts 0-3 are in the first warp.
		// Counts 4-7 are in the second warp.
		// Each stream begins on a different bank using this indexing.

		volatile uint* scan2 = predInc_shared +
			StreamLen * tid + tid / StreamsPerWarp;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// write the end-of-stream total, then perform a parallel scan.
		volatile uint* scan3 = parallelScan_shared + tid;
		
		// Add more spacing for the second warp.
		if(tid >= WARP_SIZE) scan3 += 48;

		scan3[-16] = 0;

		// For warp 0: x0 = (0, 2), x1 = (1, 3)
		// For warp 1: x0 = (4, 6), x1 = (5, 7)
		x0 = prmt(x, 0, 0x4240);
		x1 = prmt(x, 0, 0x4341);

		// Keep a copy of these unpacked terms to subtract from the inclusive
		// prefix sums to get exclusive prefix sums in the reduction array.
		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE] = x1;

		// Each thread perfoms a single parallel scan over two warps of data.
		// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 | 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
		// 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 | 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE - offset];
			x0 += y0;
			x1 += y1;
			if(i < LOG_WARP_SIZE - 1) {
				scan3[0] = x0;
				scan3[WARP_SIZE] = x1;
			}
		}

		// Add the even sums into the odd sums.
		x1 += x0;
		scan3[WARP_SIZE] = x1;

		// Warp 0 adds the term 1 total to sums 2 and 3.
		// Warp 1 adds the term 5 total to sums 6 and 7.
		uint last = scan3[(2 * WARP_SIZE - 1) - lane];
		x0 += last<< 16;
		x1 += last<< 16;

		// Store the total count for 3. This gets added into all prefix sums 
		// for elements 4 - 7.
		if(WARP_SIZE - 1 == tid) {
			uint total3 = prmt(x1, 0, 0x3232);
			parallelScan_shared[-16] = total3;
		}

		// Subtract the stream totals from the inclusive scans for the exclusive
		// scans.
		x0 -= sum0;
		x1 -= sum1;
	}
	__syncthreads();

	if(tid < 2 * WARP_SIZE) {
		if(tid >= WARP_SIZE) {
			// Get the inclusive scan through element 3.
			uint total3 = parallelScan_shared[-16];
			x0 += total3;
			x1 += total3;
		}
		uint low = prmt(x0, x1, 0x6240);
		uint high = prmt(x0, x1, 0x7351);

		parallelScan_shared[tid] = low;
		parallelScan_shared[2 * WARP_SIZE + tid] = high;
	}	
	__syncthreads();

	predInc.x = scan[0];
	predInc.y = scan[ScanSize];

	offsetsPacked.x += prmt(predInc.x, predInc.y, bucketsPacked.x);
	offsetsPacked.y += prmt(predInc.x, predInc.y, bucketsPacked.y);

	uint4 scanOffsets;
	scan = parallelScan_shared + tid / StreamLen;
	
	// Fill scanOffsets like this:
	// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L), (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H)
	scanOffsets.x = scan[0 * WARP_SIZE];
	scanOffsets.y = scan[1 * WARP_SIZE];
	scanOffsets.z = scan[2 * WARP_SIZE];
	scanOffsets.w = scan[3 * WARP_SIZE];

	return scanOffsets;
}


// scanOffsets are packed offsets for the first value of each bucket within the
// warp. They are split into high and low bytes, and packed like this:
// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L), (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H).
// bucketsPacked.x holds the first 4 buckets in the first 4 nibbles
// bucketsPacked.y holds the next 4 buckets in the first 4 nibbles
DEVICE void SortScatter3_8(uint4 scanOffsets, uint2 bucketsPacked, 
	uint2 localOffsets, uint scatter[4]) {

	// use the first 4 buckets (packed into the four nibbles of bucketsPacked.x)
	// to gather the corresponding offsets from the scan terms
	uint scan1Low = prmt(scanOffsets.x, scanOffsets.y, bucketsPacked.x);
	uint scan1High = prmt(scanOffsets.z, scanOffsets.w, bucketsPacked.x);

	// interleave the values together into packed WORDs
	// add the offsets for each value within the warp to the warp offsets
	// within the block
	scatter[0] = prmt(scan1Low, scan1High, 0x5140) + 
		ExpandUint8Low(localOffsets.x);
	scatter[1] = prmt(scan1Low, scan1High, 0x7362) + 
		ExpandUint8High(localOffsets.x);

	// Repeat the above instructions for values 4-7.
	uint scan2Low = prmt(scanOffsets.x, scanOffsets.y, bucketsPacked.y);
	uint scan2High = prmt(scanOffsets.z, scanOffsets.w, bucketsPacked.y);

	scatter[2] = prmt(scan2Low, scan2High, 0x5140) +
		ExpandUint8Low(localOffsets.y);
	scatter[3] = prmt(scan2Low, scan2High, 0x7362) + 
		ExpandUint8High(localOffsets.y);
}

