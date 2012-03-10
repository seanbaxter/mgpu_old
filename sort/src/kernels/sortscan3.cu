#pragma once

// Load 8 fused keys and produce a bucket total and a key offset within
// the thread-bucket. Extracts the 3-bit digit from each key and returns the
// uint2 bucketsPacked set of prmt masks (i.e. each of the 3-bit digits, packed
// into nibbles in the low 16 bits of .x and .y). offsetsPacked are the 
// thread-local offsets for each of the encountered keys packed into bytes.
// We can use the nibble-packed bucket indices as masks for prmt to gather the
// high and low bytes of each scan offset for each digit and add component-wise
// into offsetsPacked.

// bucketsPacked: nibble-packed digit codes for each key. .x has first four
//	  keys and .y has next four keys. Most sig 16 bits are not used.
// offsetsPacked: byte-packed thread-local offsets for each key.
// return value: nibble-packed totals for each digit. Note this overflows with
// 16 values/thread.
DEVICE uint ComputePrmtCodes(const uint* keys, uint bit, uint2& bucketsPacked, 
	uint2& offsetsPacked) {

	// predInc is the set of counts of encountered digits.
	uint predInc = 0;

	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		uint digit = bfe(keys[v], bit, 3);
		uint shift = 4 * digit;

		// Insert the number of already encountered instances of this digit
		// to offsetsPacked.
		uint encountered = predInc>> shift;

		if(0 == v) {
			predInc = 1<< shift;
			offsetsPacked.x = 0;
			bucketsPacked.x = digit;
		} else if(v < 4) {
			offsetsPacked.x = bfi(offsetsPacked.x, encountered, 8 * v, 4);
			bucketsPacked.x = bfi(bucketsPacked.x, digit, 4 * v, 4);
		} else if(4 == v) {
			offsetsPacked.y = 0x0f & encountered;
			bucketsPacked.y = digit;
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, encountered, 8 * (v - 4), 4);
			bucketsPacked.y = bfi(bucketsPacked.y, digit, 4 * (v - 4), 4);
		}
		if(v) predInc = shl_add(1, shift, predInc);
	}

	return predInc;
}


////////////////////////////////////////////////////////////////////////////////
// MultiScan3BitPrmt

DEVICE uint4 MultiScan3BitPrmt(uint tid, uint2 digitCounts, uint numThreads,
	uint2& threadOffset, uint* scratch_shared) {

	const int NumWarps = numThreads / WARP_SIZE;

	// Store two values per thread. These are 8 counters packed into 2 ints.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* byteScan_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = 2 * NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;
	
	volatile uint* parallelScan_shared = byteScan_shared + 2 * ScanSize;
	
	uint warp = tid / WARP_SIZE;

	// The first half-warp sums counts 0-3, the second half-warp sums counts
	// 4-7.
	volatile uint* scan = byteScan_shared + tid + warp;
	scan[0] = digitCounts.x;
	scan[ScanSize] = digitCounts.y;
	__syncthreads();

	if(tid < WARP_SIZE) {
		// Counts 0-3 are in the first half warp.
		// Counts 4-7 are in the second half warp.
		// Each stream begins on a different bank using this indexing.

		volatile uint* byteScan = byteScan_shared +
			StreamLen * tid + tid / StreamsPerWarp;

		// Run an inclusive scan.
		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			x += byteScan[i];

			// Only write back if the inclusive scan is different.
			if(i) byteScan[i] = x;
		}

		// There are 32 byte-packed digit totals (every StreamLen elements in
		// byteScan_shared, plus a stride). We need to unpack these into shorts
		// and parallel warp scan.
		
		// Load index1 and index1 + 1. These are redundant loads, as threads 0
		// and 8 load the same values, threads 1 and 9 load the same, etc.
		// However the even quarter-warps process the even digits and the odd
		// quarter-warps process the odd digits.
		uint index1 = 16 & tid;
		index1 += 2 * (7 & tid);
		index1 *= StreamLen;

		uint a = byteScan_shared[index1];
		uint b = byteScan_shared[index1 + 1];

		// threads 0-7 and 16-23 grab digits (0, 2) and (4, 6).
		// threads 8-15 and 24-31 grab digits (1, 3) and (5, 7).
		uint mask = (8 & tid) ? 0x4341 : 0x4240;
		uint A = prmt(a, 0, mask);
		uint B = prmt(b, 0, mask);

		// Accelerate the parallel warp scan by scanning the reduction of A and
		// B.
		x = A + B;
		uint sum = x;

		volatile uint* parallelScan = parallelScan_shared + tid + 16;
		parallelScan[-16] = 0;
		parallelScan[0] = x;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			x += parallelScan[-offset];
			parallelScan[0] = x;
		}

		// Run the fixup code.
		// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
		// 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7

		uint total13 = parallelScan_shared[15];
		uint total57 = parallelScan_shared[31];

		// For both left and right hand sides, shift the 1s total into the high
		// bits to add into the bottom row.
		uint inc;
		if(tid < WARP_SIZE / 2) {
			// For the left half, add the 1s total to the bottom row.
			inc = total13<< 16;
		} else {
			// For the right half, add the 3s total to the top row and the 5s
			// total to the bottom row.
			inc = prmt(total13, total57, 0x5432);
		}
		x += inc;

		// Subtract out the pre-scanned element count to get an exclusive scan.
		x -= sum;

		// Add in the unpacked A term to get the second scan element.
		uint scanA = x;
		uint scanB = x + A;

		// Store back to parallelScan_shared.
		uint index = 2 * tid;
		index += index / WARP_SIZE;
		parallelScan_shared[index] = scanA;
		parallelScan_shared[index + 1] = scanB;

		// Read a pair of packed shorts and store back a vector of low bytes and
		// a vector of high bytes.
		uint halfWarp = 15 & tid;
		uint offset = (tid < WARP_SIZE / 2) ? 0 : 33;
		offset += halfWarp;
		uint even = parallelScan_shared[offset];
		uint odd = parallelScan_shared[offset + 16];

		uint low = prmt(even, odd, 0x6240);
		uint high = prmt(even, odd, 0x7351);
		
		// Store back to the same offsets in shared mem.
		parallelScan_shared[offset] = low;
		parallelScan_shared[offset + 16] = high;
	}
	__syncthreads();

	// Read out the byte-packed digit counts for this thread. Subtract out 
	// digitCounts to make an exclusive scan.
	threadOffset.x = scan[0] - digitCounts.x;
	threadOffset.y = scan[ScanSize] - digitCounts.y;

	// Read out the de-interleaved short packed block scans for each digit.
	uint4 scanOffsets;
	scan = parallelScan_shared + tid / StreamLen;
	
	// Fill scanOffsets like this:
	// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L), (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H)
	scanOffsets.x = scan[0];
	scanOffsets.y = scan[33];
	scanOffsets.z = scan[16];
	scanOffsets.w = scan[33 + 16];
	
	return scanOffsets;
}


////////////////////////////////////////////////////////////////////////////////
// Get3BitPrmtScatterOffsets

// Compute packed scatter offsets for an array of 8, 16, or 24 keys. Runs just
// a single multiscan for any of these configurations.
template<int ValuesPerThread>
DEVICE2 void Get3BitPrmtScatterOffsets(const uint* keys, uint bit, 
	uint* scratch_shared, uint* packedScatter) {

	// 



}
DEVICE uint ComputePrmtCodes(const uint* keys, uint bit, uint2& bucketsPacked, 
	uint2& offsetsPacked) {

	// predInc is the set of counts of encountered digits.
	uint predInc = 0;

	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		uint digit = bfe(keys[v], bit, 3);
		uint shift = 4 * digit;

		// Insert the number of already encountered instances of this digit
		// to offsetsPacked.
		uint encountered = predInc>> shift;

		if(0 == v) {
			predInc = 1<< shift;
			offsetsPacked.x = 0;
			bucketsPacked.x = digit;
		} else if(v < 4) {
			offsetsPacked.x = bfi(offsetsPacked.x, encountered, 8 * v, 4);
			bucketsPacked.x = bfi(bucketsPacked.x, digit, 4 * v, 4);
		} else if(4 == v) {
			offsetsPacked.y = 0x0f & encountered;
			bucketsPacked.y = digit;
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, encountered, 8 * (v - 4), 4);
			bucketsPacked.y = bfi(bucketsPacked.y, digit, 4 * (v - 4), 4);
		}
		if(v) predInc = shl_add(1, shift, predInc);
	}

	return predInc;
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
}*/



/*
DEVICE uint4 MultiScan3BitPrmt(uint tid, uint2 digitCounts, uint numThreads,
	uint2& threadOffset, uint* scratch_space) {





// TDOO: use a single warp in the reduction. This doubles the sequential
// scan legnth from 4 elements to 8, but slightly simplifies the parallel scan
// (11 operations) and lets it execute on just one warp rather than 2.

// Should be a net savings of 11 ld/st pairs, at the cost of lower effective
// occupancy.


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
}*/
