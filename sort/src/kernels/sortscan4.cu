#pragma once

// MultiScan4_1Warp operates safely on blocks with up to 1024 values.
DEVICE void MultiScan4_1Warp(uint tid, uint4 predInc, uint numThreads,
	uint bucketsPacked, uint4& localOffsets2, uint4& offsetsLow,
	uint4& offsetsHigh, uint* scratch_shared, uint* debug_global) {
	
	const int NumWarps = numThreads / WARP_SIZE;

	const int ScanSize = numThreads + NumWarps;
	volatile* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = 4 * NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;

	// const ParallelScanSize = 2 * WARP_SIZE + 16 + 32
	volatile uint* parallelScan_shared = predInc_shared + 4 * ScanSize + 16;

	uint warp = tid / WARP_SIZE;

	// Quarter-warp 0 sums counts 0-3.
	// Quarter-warp 1 sums counts 4-7.
	// Quarter-warp 2 sums counts 8-11.
	// Quarter-warp 3 sums counts 12-15;
	volatile uint* scan = predInc_shared + tid + warp;

	// predInc.x = (0, 1, 2, 3)
	// predInc.y = (4, 5, 6, 7)
	// predInc.z = (8, 9, 10, 11)
	// predInc.w = (12, 13, 14, 15).
	scan[0 * ScanSize] = predInc.x;
	scan[1 * ScanSize] = predInc.y;
	scan[2 * ScanSize] = predInc.z;
	scan[3 * ScanSize] = predInc.w;
	__syncthreads(); 

	if(tid < 2 * WARP_SIZE) {
		volatile uint* scan2 = predInc_shared + 
			StreamLen * tid + tid / StreamsPerWarp;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// Write the end-of-stream total, then perform a parallel scan.
		volatile uint* warpStart = parallelScan_shared;
		if(tid >= WARP_SIZE) warpStart += 48;

		// Add more spacing for the second warp.
		volatile uint* scan3 = warpStart + tid;
		if(16 & tid) scan3 += 16;

		scan3[-16] = 0;

		// For tid  0-15: x0 = (0, 2),   x1 = (1, 3) 
		// For tid 16-31: x0 = (4, 6),   x1 = (5, 7)
		// For tid 32-47: x0 = (8, 10),  x1 = (9, 11)
		// For tid 48-63: x0 = (12, 14), x1 = (13, 15)
		x0 = prmt(x, 0, 0x4240);
		x1 = prmt(x, 0, 0x4341);

		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE / 2] = x1;

		// 0  0  1  1  4  4  5  5 |  8  8  9  9 12 12 13 13
		// 2  2  3  3  6  6  7  7 | 10 10 11 11 14 14 15 15

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< offset;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE / 2 - offset];
			x0 += y0;
			x1 += y1;
			
			scan3[0] = x0;
			scan3[WARP_SIZE / 2] = x1;
		}

		// We've scanned the two halves. 
		
		// For warp 0: add last 1 to all on bottom. Add last 5 to 6 and 7.
		// For warp 1: add last 9 to all to bottom. Add last 13 to 14 and 15.

		// mid holds the last value with elements (1, 3).
		// rigth holds the last value with elements (5, 7).
		uint mid = warpStart[WARP_SIZE / 2 - 1];
		uint right = warpStart[WARP_SIZE - 1];

		// Add 1 to the bottom row.
		uint inc = mid<< 16;

		// If we're in the right half of the warp...
		if(16 & tid) {
			// Add 3 to all elements.
			inc += prmt(mid, 0, 0x3232);

			// Add 5 to the bottom row.
			inc += right<< 16;
		}

		x0 += inc;
		x1 += inc;

		// Store the total count for 7. This gets added into all prefix sums
		// for elements 8-15.
		if(WARP_SIZE - 1 == tid) {
			uint total7 = prmt(x1, 0, 0x3232);
			parallelScan_shared[-16] = total7;
		}

		// Subtract the stream totals from the inclusive scans.
		x0 -= sum0;
		x1 -= sum1;
	}
	__syncthreads();

	if(tid < 2 * WARP_SIZE) {
		if(tid >= WARP_SIZE) {
			// Get the inclusive scan through element 7.
			uint total7 = parallelScan_shared[-16];
			x0 += total7;
			x1 += total7;
		}

		// Split the counters into low and high bytes.
		uint low = prmt(x0, x1, 0x6240);
		uint high = prmt(x0, x1, 0x7351);

		volatile uint* offsets = parallelScan_shared;
		if(16 & tid) offsets += 16;

		offsets[tid] = low;
		offsets[WARP_SIZE / 2 + tid] = high;
	}
	__syncthreads();

	// Read out the packed values.

	uint scan0 = scan[0 * WARP_SIZE];
	uint scan1 = scan[1 * WARP_SIZE];
	uint scan2 = scan[2 * WARP_SIZE];
	uint scan3 = scan[3 * WARP_SIZE];
	uint scan4 = scan[4 * WARP_SIZE];
	uint scan5 = scan[5 * WARP_SIZE];
	uint scan6 = scan[6 * WARP_SIZE];
	uint scan7 = scan[7 * WARP_SIZE];
}


/*
DEVICE void MultiScan4_2Warp(uint tid, uint4 predInc, uint numThreads,
	uint bucketsPacked, uint2& offsetsPacked, uint4& localOffsets2, 
	uint4& offsetsLow, uint4& offsetsHigh, uint* scratch_shared, 
	uint* debug_global) {
	
	const int NumWarps = numThreads / WARP_SIZE;

	const int ScanSize = numThreads + NumWarps;
	volatile* predInc_shared = (volatile uint*)scratch_shared;

	const int StreamLen = 2 * NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;

	volatile uint* scan = predInc_shared + tid + warp;

	// predInc.x = (0, 1, 2, 3)
	// predInc.y = (4, 5, 6, 7)
	// predInc.z = (8, 9, 10, 11)
	// predInc.w = (12, 13, 14, 15).

	scan[0 * ScanSize] = predInc.x;
	scan[1 * ScanSize] = predInc.y;
	scan[2 * ScanSize] = predInc.z;
	scan[3 * ScanSize] = predInc.w;
	__syncthreads(); 

	uint x0, x1;
	if(tid < 2 * WARP_SIZE) {
		volatile uint* scan2 = predInc_shared + 
			StreamLen * tid + tid / StreamsPerWarp;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// Write the end-of-stream total, then perform a parallel scan.
		volatile uint* warpStart = parallelScan_shared;
		if(tid >= WARP_SIZE) warpStart += 48;

		// Add more spacing for the second warp.
		volatile uint* scan3 = warpStart + tid;
		if(16 & tid) scan3 += 16;

		scan3[-16] = 0;

		// For tid  0-15: x0 = (0, 2),   x1 = (1, 3) 
		// For tid 16-31: x0 = (4, 6),   x1 = (5, 7)
		// For tid 32-47: x0 = (8, 10),  x1 = (9, 11)
		// For tid 48-63: x0 = (12, 14), x1 = (13, 15)
		x0 = prmt(x, 0, 0x4240);
		x1 = prmt(x, 0, 0x4341);

		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE / 2] = x1;

		// 0  0  1  1  4  4  5  5 |  8  8  9  9 12 12 13 13
		// 2  2  3  3  6  6  7  7 | 10 10 11 11 14 14 15 15

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< offset;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE / 2 - offset];
			x0 += y0;
			x1 += y1;
			
			scan3[0] = x0;
			scan3[WARP_SIZE / 2] = x1;
		}

		// We've scanned the two halves. 
		
		// For warp 0: add last 1 to all on bottom. Add last 5 to 6 and 7.
		// For warp 1: add last 9 to all to bottom. Add last 13 to 14 and 15.

		// mid holds the last value with elements (1, 3).
		// rigth holds the last value with elements (5, 7).
		uint mid = warpStart[WARP_SIZE / 2 - 1];
		uint right = warpStart[WARP_SIZE - 1];

		// Add 1 to the bottom row.
		uint inc = mid<< 16;

		// If we're in the right half of the warp...
		if(16 & tid) {
			// Add 3 to all elements.
			inc += prmt(mid, 0, 0x3232);

			// Add 5 to the bottom row.
			inc += right<< 16;
		}

		x0 += inc;
		x1 += inc;

		// Store the total count for 7. This gets added into all prefix sums
		// for elements 8-15.
		if(WARP_SIZE - 1 == tid) {
			uint total7 = prmt(x1, 0, 0x3232);
			parallelScan_shared[-16] = total7;
		}

		// Subtract the stream totals from the inclusive scans.
		x0 -= sum0;
		x1 -= sum1;
	}
	__syncthreads();

	if(tid < 2 * WARP_SIZE) {
		if(tid >= WARP_SIZE) {
			// Get the inclusive scan through element 7.
			uint total7 = parallelScan_shared[-16];
			x0 += total7;
			x1 += total7;
		}

		// Split the counters into low and high bytes.
		uint low = prmt(x0, x1, 0x6240);
		uint high = prmt(x0, x1, 0x7351);

		volatile uint* offsets = parallelScan_shared;
		if(16 & tid) offsets += 16;

		offsets[tid] = low;
		offsets[WARP_SIZE / 2 + tid] = high;
	}
	__syncthreads();

	// Read out the packed values.

	uint scan0 = scan[0 * WARP_SIZE];
	uint scan1 = scan[1 * WARP_SIZE];
	uint scan2 = scan[2 * WARP_SIZE];
	uint scan3 = scan[3 * WARP_SIZE];
	uint scan4 = scan[4 * WARP_SIZE];
	uint scan5 = scan[5 * WARP_SIZE];
	uint scan6 = scan[6 * WARP_SIZE];
	uint scan7 = scan[7 * WARP_SIZE];
}
*/

DEVICE void SortScatter4_8(uint4 offsetsLow, uint4 offsetsHigh, uint buckets,
	uint2 localOffsets, uint4 localOffsets2, uint scatter[4]) {

	// offsetsLow and offsetsHigh hold (0 - 15)L and (0 - 15)H. prmt is used to
	// grab the required digits for each value.

	// buckets has a digit code packed into each nibble. The digit code takes 
	// all four bits of the nibble, so cannot be used with prmt directly.

	// localOffsets is the set of thread-local offsets for each value with the
	// digits already having been selected. These do not need a prmt operation.
	// They are packed into bytes.

	// localOffsets2 is the set of offsets to be added to both offsetsLow/High
	// and localOffsets. It is stored per-digit so must be removed with prmt.


	// Mask out the high bit of each nibble. This lets us use prmt to gather.
	uint prmtMasked = 0x77777777 & buckets;
	uint prmtGather = 0x32103210 + ((0x88888888 & buckets)>> 1);
	
	// Compute the offsets for the first four values.
	// Update localOffsets, which is organized by value (not by radix digit) and
	// byte-packed. We can add in the scan offsets without risk of overflow.
	uint scanLeft0 = prmt(localOffsets2.x, localOffsets2.y, prmtMasked);
	uint scanRight0 = prmt(localOffsets2.z, localOffsets2.w, prmtMasked);
	uint scan0 = prmt(scanLeft0, scanRight0, prmtGather);
	localOffsets.x += scan0;

	// Run a 16-way prmt to gather the low and high bytes of all 8 values from
	// offsetsLow and offsetsHigh.
	uint scan0LowLeft = prmt(offsetsLow.x, offsetsLow.y, prmtMasked);
	uint scan0LowRight = prmt(offsetsLow.z, offsetsLow.w, prmtMasked);
	uint scan0Low = prmt(scan0LowLeft, scan0LowRight, prmtGather);

	uint scan0HighLeft = prmt(offsetsHigh.x, offsetsHigh.y, prmtMasked);
	uint scan0HighRight = prmt(offsetsHigh.z, offsetsHigh.w, prmtMasked);
	uint scan0High = prmt(scan0HighLeft, scan0HighRight, prmtGather);

	scatter[0] = prmt(scan0Low, scan0High, 0x5140) + 
		ExpandUint8Low(localOffsets.x);
	scatter[1] = prmt(scan0Low, scan0High, 0x7362) +
		ExpandUint8High(localOffsets.x);

	// Compute the offsets for the last four values.
	prmtMasked >>= 16;
	prmtGather >>= 16;

	uint scanLeft1 = prmt(localOffsets2.x, localOffsets2.y, prmtMasked);
	uint scanRight1 = prmt(localOffsets2.z, localOffsets2.w, prmtMasked);
	uint scan1 = prmt(scanLeft1, scanRight1, prmtGather2);
	localOffsets.y += scan1;

	uint scan1LowLeft = prmt(offsetsLow.x, offsetsLow.y, prmtMasked);
	uint scan1LowRight = prmt(offsetsLow.z, offsetsLow.w, prmtMasked);
	uint scan1Low = prmt(scan1LowLeft, scan1LowRight, prmtGather);

	uint scan1HighLeft = prmt(offsetHigh.x, offsetsHigh.y, prmtMasked);
	uint scan1HighRight = prmt(offsetHigh.z, offsetsHigh.w, prmtMasked);
	uint scan1High = prmt(scan1HighLeft, scan1HighRight, prmtGahter);

	scatter[2] = prmt(scan1Low, scan1High, 0x5140) +
		ExpandUint8Low(localOffsets.y);
	scatter[3] = prmt(scan1Low, scan1High, 0x7362) +
		ExpandUint8High(localOffsets.y);
}

