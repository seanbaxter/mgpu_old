#pragma once

DEVICE void MultiScan4(uint tid, uint4 predInc, uint numThreads,
	uint bucketsPacked, uint2& offsetsPacked, uint* scratch_shared,
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
	scan[0] = predInc.x;
	scan[ScanSize] = predInc.y;
	scan[2 * ScanSize] = predInc.z;
	scan[3 * ScanSize = predInc.w;
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
		volatile uint* scan3 = parallelScan_shared + tid;

		// Add more spacing for the second warp.
		if(tid >= WARP_SIZE) scan3 += 48;

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
		scan3[WARP_SIZE] = x1;

		// 0 0 0 0 1 1 1 1 






	}

	uint threadScan0;			// bytes 0 - 3
	uint threadScan1;			// bytes 4 - 7
	uint threadScan2;			// bytes 8 - 11
	uint threadScan3;			// bytes 12- 15





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
	uint prmtMasked = 0x77777777 & bucketsPacked;
	uint prmtGather = 0x32103210 + ((0x88888888 & bucketsPacked)>> 1);
	
	// Compute the offsets for the first four values.
	// Update localOffsets, which is organized by value (not by radix digit) and
	// byte-packed. We can add in the scan offsets without risk of overflow.
	uint scanLeft0 = prmt(threadScan0, threadScan1, prmtMasked);
	uint scanRight0 = prmt(threadScan2, threadScan3, prmtMasked);
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

	uint scanLeft1 = prmt(threadScan0, threadScan1, prmtMasked2);
	uint scanRight1 = prmt(threadScan2, threadScan3, prmtMasked2);
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

