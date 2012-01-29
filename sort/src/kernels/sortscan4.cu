#pragma once

// MultiScan4_1Warp operates safely on blocks with up to 1024 values.
DEVICE void MultiScan4_1Warp(uint tid, uint4& predInc, uint numThreads,
	uint bucketsPacked, uint4& offsetsLow, uint4& offsetsHigh, 
	uint* scratch_shared, uint* debug_global) {
	
	const int NumWarps = numThreads / WARP_SIZE;

	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements.
	const int StreamLen = 4 * NumWarps;
	const int StreamsPerWarp = WARP_SIZE / StreamLen;

	// const ParallelScanSize = 2 * WARP_SIZE + 16 + 32
	volatile uint* parallelScan_shared = predInc_shared + 4 * ScanSize + 8;

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

	if(tid < WARP_SIZE) {
		volatile uint* scan2 = predInc_shared + 
			StreamLen * tid + tid / StreamsPerWarp;

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < StreamLen; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// 0 0 0 0 1 1 1 1 | 4 4 4 4 5 5 5 5 | 8 8 8 8 9 9 9 9 | c c c c d d d d 
		// 2 2 2 2 3 3 3 3 | 6 6 6 6 7 7 7 7 | a a a a b b b b | e e e e f f f f

		// We perform 4 independent parallel scans with each thread maintaining
		// two values.
		// tid 0  ->  0 ( 0),  8 ( 8)			start at tid + 0
		// tid 8  -> 24 (24), 32 ( 0)			start at tid + 16
		// tid 16 -> 48 (16), 56 (24)			start at tid + 32
		// tid 24 -> 72 ( 8), 80 (16)			start at tid + 48

		volatile uint* scan3 = parallelScan_shared + tid + ((24 & tid)<< 1);
		scan3[-8] = 0;

		// For tid 0-7:  x0 = (0, 2), x1 = (1, 3)
		// For tid 8-15: x0 = (4, 6), x1 = (5, 7)
		// For tid 16-23: x0 = (8, 10), x1 = (9, 11)
		// For tid 24-31: x0 = (12, 14), x1 = (13, 15)
		uint x0 = prmt(x, 0, 0x4240);
		uint x1 = prmt(x, 0, 0x4341);

		// Keep a copy of these unpacked terms to subtract from the inclusive
		// prefix sums to get exclusive prefix sums in the reduction array.
		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE / 4] = x1;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE - 2; ++i) {
			int offset = 1<< i;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE / 4 - offset];
			x0 += y0;
			x1 += y1;

			if(i < LOG_WARP_SIZE - 3) {
				scan3[0] = x0;
				scan3[WARP_SIZE / 4] = x1;
			}
		}
		x1 += x0;
		scan3[WARP_SIZE / 4] = x1;

		// Pull the four values from the end of each section.
		// last0 = (a, b)
		// last1 = (c, d)
		// last2 = (e, f)
		// last3 = (g, h)
		uint last0 = parallelScan_shared[15];
		uint last1 = parallelScan_shared[39];
		uint last2 = parallelScan_shared[63];
		uint last3 = parallelScan_shared[87];

		// inc = (0, a)
		uint inc = last0<< 16;			

		if(tid >= 8) {
			// inc = (a, a + b)
			inc += last0;	

			// inc = (a + b, a + b + c)			
			inc = prmt(inc, 0, 0x3232) + (last1<< 16);
		}
		if(tid >= 16) {
			// inc = (a + b + c, a + b + c + d)
			inc += last1;

			// inc = (a + b + c + d, a + b + c + d + e)
			inc = prmt(inc, 0, 0x3232) + (last2<< 16);
		}
		if(tid >= 24) {
			// inc = (a + b + c + d + e, a + b + c + d + e + f)
			inc += last2;

			// inc = (a + b + c + d + e + f, a + b + c + d + e + f + g)
			inc = prmt(inc, 0, 0x3232) + (last3<< 16);
		}
		x0 += inc;
		x1 += inc;

		// Subtract the stream totals from the inclusive scans.
		x0 -= sum0;
		x1 -= sum1;

		// Split the offsets into low and high bytes.
		uint low = prmt(x0, x1, 0x6240);
		uint high = prmt(x0, x1, 0x7351);

		parallelScan_shared[tid] = low;
		parallelScan_shared[WARP_SIZE + tid] = high;
	}
	__syncthreads();

	predInc.x = scan[0 * ScanSize];
	predInc.y = scan[1 * ScanSize];
	predInc.z = scan[2 * ScanSize];
	predInc.w = scan[3 * ScanSize];

	scan = parallelScan_shared + tid / StreamLen;
	offsetsLow.x = scan[0 * WARP_SIZE / 4];
	offsetsLow.y = scan[1 * WARP_SIZE / 4];
	offsetsLow.z = scan[2 * WARP_SIZE / 4];
	offsetsLow.w = scan[3 * WARP_SIZE / 4];
	offsetsHigh.x = scan[4 * WARP_SIZE / 4];
	offsetsHigh.y = scan[5 * WARP_SIZE / 4];
	offsetsHigh.z = scan[6 * WARP_SIZE / 4];
	offsetsHigh.w = scan[7 * WARP_SIZE / 4];
}


DEVICE void SortScatter4_8(uint4 offsetsLow, uint4 offsetsHigh, uint buckets,
	uint4 predInc, uint2 localOffsets, uint scatter[4]) {

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
	uint scanLeft0 = prmt(predInc.x, predInc.y, prmtMasked);
	uint scanRight0 = prmt(predInc.z, predInc.w, prmtMasked);
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

	uint scanLeft1 = prmt(predInc.x, predInc.y, prmtMasked);
	uint scanRight1 = prmt(predInc.z, predInc.w, prmtMasked);
	uint scan1 = prmt(scanLeft1, scanRight1, prmtGather);
	localOffsets.y += scan1;

	uint scan1LowLeft = prmt(offsetsLow.x, offsetsLow.y, prmtMasked);
	uint scan1LowRight = prmt(offsetsLow.z, offsetsLow.w, prmtMasked);
	uint scan1Low = prmt(scan1LowLeft, scan1LowRight, prmtGather);

	uint scan1HighLeft = prmt(offsetsHigh.x, offsetsHigh.y, prmtMasked);
	uint scan1HighRight = prmt(offsetsHigh.z, offsetsHigh.w, prmtMasked);
	uint scan1High = prmt(scan1HighLeft, scan1HighRight, prmtGather);

	scatter[2] = prmt(scan1Low, scan1High, 0x5140) +
		ExpandUint8Low(localOffsets.y);
	scatter[3] = prmt(scan1Low, scan1High, 0x7362) +
		ExpandUint8High(localOffsets.y);
}

