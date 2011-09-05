

////////////////////////////////////////////////////////////////////////////////
// MultiScan3

// Compute the stream length so that each thread computes just one stream.
// Splitting these inclusive offsets from bytes into words will then require
// parallel scans over two warps of data.

// A block of 8 warps requires scan of 2 streams per channel (0-3, 4-7) per
// warp, with a stream length of 16. Each lane writes to bank lane + warp. 
// If 8 == StreamLength:
// tid 0 sums 0-7		(bank 0)
// tid 1 sums 8-15		(bank 8)
// tid 2 sums 16-23		(bank 16)
// tid 3 sums 24-31		(bank 24)
// tid 4 sums 33-40		(bank 1)
// tid 5 sums 41-48		(bank 9)
// tid 6 sums 49-57		(bank 17)
// tid 7 sums 58-65		(bank 25)
// tid 8 sums 67-74		(bank 3)...

#define SCAN_SIZE_3 (NUM_THREADS + (NUM_THREADS / WARP_SIZE))
#define STREAMS_PER_WARP_3 (WARP_SIZE / (2 * NUM_WARPS))
#define STREAM_LENGTH_3 (WARP_SIZE / STREAMS_PER_WARP_3)

#define predInc3_shared reduction_shared
#define reduction3_shared (predInc3_shared + 2 * SCAN_SIZE_3)
#define parallelScan3_shared (scattergather_shared + 16)

// For 128 threads:
// STREAMS_PER_WARP = 4
// STREAM_LENGTH = 8

// For 256 threads:
// STREAMS_PER_WARP = 2
// STREAM_LENGTH = 16

// The scan space for predInc is 2 * (NUM_THREADS + (NUM_THREADS / WARP_SIZE))
// The scan space for the streams is exactly 64. Each thread in the warp (32)
// deos a sequential scan, and at the end, these values are unpacked to
// accomodate more than 255 offsets, making 64 values.


// Use a single warp for the multi-scan. For blocks of 128 threads, the stream 
// length is only 8 elements. For blocks of 256 threads, the stream length is 16
// elements. No explicit synchronization is required. This long run of
// instructions on a single warp may cause pipeline bubbles depending on 
// configuration.

// If we want to prepare the expanded scatter list in parallel with the
// single-warp scan, pass numBuckets for prepareCoalesced.

DEVICE uint4 MultiScan3(uint tid, uint2 predInc, uint2 bucketsPacked,
	uint2& offsetsPacked, uint numTransBuckets, volatile uint* compressed,
	volatile uint* uncompressed, uint* debug_global) {

	// The first half-warp sums counts 0-3, the second half-warp sums counts
	// 4-7.
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	volatile uint* scan = predInc3_shared + (tid + warp);
	scan[0] = predInc.x;
	scan[SCAN_SIZE_3] = predInc.y;
	__syncthreads();

	// Perform sequential scan over the byte-packed counts.
	// The sequential operation exhibits very little ILP (only the addition and
	// next LDS can be run in parallel). This is the major pipeline bottleneck
	// for the kernel. We need to launch enough blocks to hide this latency.

	// The parallel scan in the second half is executed on two warps of data, in
	// parallel. The ILP here is 2. It is still a major bottleneck, but 
	// effectively will run twice as fast as the sequential part.
	if(tid < WARP_SIZE) {
		// Counts 0-3 are in the first half of scan.
		// Counts 4-7 are in the second half of scan.
		// Each stream begins on a different bank using this indexing.
		volatile uint* scan2 = predInc3_shared + 
			(STREAM_LENGTH_3 * tid + tid / STREAMS_PER_WARP_3);
		
		uint x = 0;
		#pragma unroll
		for(int i = 0; i < STREAM_LENGTH_3; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// If tid < WARP_SIZE / 2, process counts 0-3.
		// If tid >= WARP_SIZE / 2, process counts 4-7.
		// Unpack 0-3 to low = (0, 2), high = (1, 3).
		// Unpack 4-7 to low = (4, 6), high = (5, 7).

		// Write the end-of-stream total, then perform a parallel scan.
		// Unpack the counts to tid and tid + WARP_SIZE / 2
		volatile uint* scan3 = parallelScan3_shared + tid;
		scan2 = scan3;
		if(tid >= WARP_SIZE / 2) scan2 += WARP_SIZE;
		scan2[-16] = 0;
		scan2[0] = prmt(x, 0, 0x4240);
		scan2[WARP_SIZE / 2] = prmt(x, 0, 0x4341);
	
		// Perform parallel scan over two warps of data.
		// This thread processes columns
		// tid (0+2 or 1+3) and 
		// tid + WARP_SIZE (4+6 or 5+7).
		// 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 | 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5
		// 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 | 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7
		volatile uint* scan4 = scan3 + WARP_SIZE + 16;
		uint x0 = scan3[0];
		uint x1 = scan4[0];
		uint counts0 = x0;
		uint counts1 = x1;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			uint y0 = scan3[-offset];
			uint y1 = scan4[-offset];
			x0 += y0;
			x1 += y1;
			scan3[0] = x0;
			scan4[0] = x1;
		}
		// Add the top-right sum of each half to all bottom sums of each half.
		uint sum0 = parallelScan3_shared[WARP_SIZE - 1];
		uint sum1 = parallelScan3_shared[(WARP_SIZE + 16) + WARP_SIZE - 1];

		x0 = shl_add_c(sum0, 16, x0);
		x1 = shl_add_c(sum1, 16, x1);

		// Add the last total of the first half to all totals (both low and
		// high) of the second half.
		x1 += prmt(shl_add_c(sum0, 16, sum0), 0, 0x3232);

		// Subtract the original counts to get an exclusive sacn
		x0 -= counts0;
		x1 -= counts1;
		
		// This thread holds offsets (0, 2, 4, 6) or (1, 3, 5, 7). Switch around
		// with the other threads to get consecutive offsets.
		scan3[0] = x0;
		scan4[0] = x1;

		if(tid < WARP_SIZE / 2) {
			uint a = scan3[0];				// 0, 2
			uint b = scan3[16];				// 1, 3
			uint c = scan4[0];				// 4, 6
			uint d = scan4[16];				// 5, 7

			// output in order:
			// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L), 
			// (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H).
			reduction3_shared[2 * tid + 0] = prmt(a, b, 0x6240); 
			reduction3_shared[2 * tid + 1] = prmt(c, d, 0x6240);
			reduction3_shared[2 * tid + WARP_SIZE + 0] = prmt(a, b, 0x7351);
			reduction3_shared[2 * tid + WARP_SIZE + 1] = prmt(c, d, 0x7351);
		}
	} else if(numTransBuckets && 1 == warp)
		// Expand the transaction list in parallel with the single-warp scan.
		ExpandScatterList(lane, numTransBuckets, compressed, uncompressed,
			debug_global);
	
	__syncthreads();

	predInc.x = scan[0];
	predInc.y = scan[SCAN_SIZE_3];

	offsetsPacked.x += prmt(predInc.x, predInc.y, bucketsPacked.x);
	offsetsPacked.y += prmt(predInc.x, predInc.y, bucketsPacked.y);

	uint4 sortOffsets;
	scan = reduction3_shared + 2 * (tid / STREAM_LENGTH_3);
	
	sortOffsets.x = scan[0];
	sortOffsets.y = scan[1];
	sortOffsets.z = scan[WARP_SIZE];
	sortOffsets.w = scan[WARP_SIZE + 1];

	return sortOffsets;
}



// sortOffsets are packed offsets for the first value of each bucket within the
// warp. They are split into high and low bytes, and packed like this:
// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L), (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H).
// bucketsPacked.x holds the first 4 buckets in the first 4 nibbles
// bucketsPacked.y holds the next 4 buckets in the first 4 nibbles
DEVICE void SortScatter3_8(uint4 sortOffsets, uint2 bucketsPacked, 
	uint2 localOffsets, Values fusedKeys, uint scatter[4], uint tid) {

	// use the first 4 buckets (packed into the four nibbles of bucketsPacked.x)
	// to gather the corresponding offsets from the scan terms
	uint scan1Low = prmt(sortOffsets.x, sortOffsets.y, bucketsPacked.x);
	uint scan1High = prmt(sortOffsets.z, sortOffsets.w, bucketsPacked.x);

	// interleave the values together into packed WORDs
	// add the offsets for each value within the warp to the warp offsets
	// within the block
	scatter[0] = prmt(scan1Low, scan1High, 0x5140) + 
		ExpandUint8Low(localOffsets.x);
	scatter[1] = prmt(scan1Low, scan1High, 0x7362) + 
		ExpandUint8High(localOffsets.x);

	// Repeat the above instructions for values 4-7.
	uint scan2Low = prmt(sortOffsets.x, sortOffsets.y, bucketsPacked.y);
	uint scan2High = prmt(sortOffsets.z, sortOffsets.w, bucketsPacked.y);

	scatter[2] = prmt(scan2Low, scan2High, 0x5140) +
		ExpandUint8Low(localOffsets.y);
	scatter[3] = prmt(scan2Low, scan2High, 0x7362) + 
		ExpandUint8High(localOffsets.y);
}

