// TDOO: use a single warp in the reduction. This doubles the sequential
// scan legnth from 4 elements to 8, but slightly simplifies the parallel scan
// (11 operations) and lets it execute on just one warp rather than 2.

// Should be a net savings of 11 ld/st pairs, at the cost of lower effective
// occupancy.

#pragma once

// MultiScan3_1Warp supports blocks with up to 2048 values.
DEVICE uint4 MultiScan3_1Warp(uint tid, uint2 predInc, uint numThreads, 
	uint2 bucketsPacked, uint2& offsetsPacked, uint* scratch_shared,
	uint* debug_global) {

	const int NumWarps = numThreads / WARP_SIZE;

	// Store two values per thread. These are 8 counters packed into 2 ints.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = 2 * NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;

	// const ParallelScanSize = 2 * WARP_SIZE + 16 + 32;
	volatile uint* parallelScan_shared = predInc_shared + 2 * ScanSize + 16;

	uint warp = tid / WARP_SIZE;

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
	if(tid < WARP_SIZE) {
		// Counts 0-3 are in the first half warp.
		// Counts 4-7 are in the second half warp.
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

		// The first half-warp (tid 0:15) has elements 0-3 in x.
		// The second half-warp (tid 16:31) has elements 4-7 in x.

		// The elements are unpacked and processed in two separate arrays like
		// this:
		// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1  |  4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
		// 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3  |  6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7

		// Sufficient spacing is given before each half-warp's data so that the
		// scans are completely separate. The first half-warp of threads manage
		// slots tid and 16 + tid. To avoid bank conflicts, tid 16 must start
		// at offset 48 to prevent bank conflicts.

		// Write the end-of-stream total, then perform a parallel scan.
		volatile uint* scan3 = parallelScan_shared + tid;

		// Add more spacing for the second half-warp.
		if(tid >= WARP_SIZE / 2) scan3 += 32;

		scan3[-16] = 0;

		// For tid 0-15:  x0 = (0, 2), x1 = (1, 3)
		// For tid 16-31: x0 = (4, 6), x1 = (5, 7)
		uint x0 = prmt(x, 0, 0x4240);
		uint x1 = prmt(x, 0, 0x4341);

		// Keep a copy of these unpacked terms to subtract from the inclusive
		// prefix sums to get exclusive prefix sums in the reduction array.
		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE / 2] = x1;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE - 1; ++i) {
			int offset = 1<< i;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE / 2 - offset];
			x0 += y0;
			x1 += y1;

			if(i < LOG_WARP_SIZE - 2) {
				scan3[0] = x0;
				scan3[WARP_SIZE / 2] = x1;
			}
		}
		x1 += x0;
		scan3[WARP_SIZE / 2] = x1;

		uint midLast = parallelScan_shared[31];
		uint rightLast = parallelScan_shared[79];

		// Run the fixup code to turn these two independent parallel scans into
		// a continuous scan.
		// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1  |  4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
		// 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3  |  6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7

		uint inc = midLast<< 16;
		if(16 & tid) {
			// We're in the right half (second half-warp). Add 1 + 3 to all
			// elements and add 5 to the bottom row.
			inc += midLast;
			inc = prmt(inc, 0, 0x3232) + (rightLast<< 16);
		}
		x0 += inc;
		x1 += inc;

		// Subtract the stream totals from the inclusive scans for the exclusive
		// scans.
		x0 -= sum0;
		x1 -= sum1;

		// Split the offsets into low and high bytes.
		uint low = prmt(x0, x1, 0x6240);
		uint high = prmt(x0, x1, 0x7351);

		parallelScan_shared[tid] = low;
		parallelScan_shared[WARP_SIZE + tid] = high;
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
	scanOffsets.x = scan[0 * WARP_SIZE / 2];
	scanOffsets.y = scan[1 * WARP_SIZE / 2];
	scanOffsets.z = scan[2 * WARP_SIZE / 2];
	scanOffsets.w = scan[3 * WARP_SIZE / 2];

	return scanOffsets;
}

DEVICE uint4 MultiScan3_2Warp(uint tid, uint2 predInc, uint numThreads, 
	uint2 bucketsPacked, uint2& offsetsPacked, uint* scratch_shared,
	uint* debug_global) {

	const int NumWarps = numThreads / WARP_SIZE;

	// Store two values per thread. These are 8 counters packed into 2 ints.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;

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

		// Each thread performs a single parallel scan over two warps of data.
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

		// Subtract the stream totals from the inclusive scans.
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