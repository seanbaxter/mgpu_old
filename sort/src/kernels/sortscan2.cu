
#define SCAN_SIZE_2 (NUM_THREADS + (NUM_THREADS / WARP_SIZE))
#define STREAMS_PER_WARP_2 (WARP_SIZE / NUM_WARPS)
#define STREAM_LENGTH_2 (WARP_SIZE / STREAMS_PER_WARP_2)

#define predInc2_shared reduction_shared
#define reduction2_shared (predInc2_shared + SCAN_SIZE_2)
#define parallelScan2_shared (scattergather_shared + 16)

DEVICE uint2 MultiScan2(uint tid, uint predInc, uint numTransBuckets,
	volatile uint* compressed, volatile uint* uncompressed, 
	uint* debug_global) {

	volatile uint* scan = predInc2_shared + (tid + tid / WARP_SIZE);
	scan[0] = predInc;
	__syncthreads();

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	// Perform sequential scan over the byte-packed counts.
	// The sequential operation exhibits very little ILP (only the addition and
	// next LDS can be run in parallel). This is the major pipeline bottleneck
	// for the kernel. We need to launch enough blocks to hide this latency.

	// The parallel scan in the second half is executed on two warps of data, in
	// parallel. The ILP here is 2. It is still a major bottleneck, but 
	// effectively will run twice as fast as the sequential part.
	if(tid < WARP_SIZE) {
		// Each stream begins on a different bank using this indexing.
		volatile uint* scan2 = predInc2_shared + 
			(STREAM_LENGTH_2 * tid + tid / STREAMS_PER_WARP_2);

		uint x = 0;
		#pragma unroll
		for(int i = 0; i < STREAM_LENGTH_2; ++i) {
			uint y = scan2[i];
			scan2[i] = x;
			x += y;
		}

		// Write the end-of-stream total, then perform a parallel scan.
		volatile uint* scan3 = parallelScan2_shared + tid;	
		scan3[-16] = 0;

		uint x0 = prmt(x, 0, 0x4240);
		uint x1 = prmt(x, 0, 0x4341);

		uint sum0 = x0;
		uint sum1 = x1;

		scan3[0] = x0;
		scan3[WARP_SIZE] = x1;

		// Perform a single parallel scan over two warps of data.
		// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
		// 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			uint y0 = scan3[-offset];
			uint y1 = scan3[WARP_SIZE - offset];
			x0 += y0;
			x1 += y1;
			scan3[0] = x0;
			scan3[WARP_SIZE] = x1;
		}
		x1 += x0;
		scan3[WARP_SIZE] = x1;

		// Add the 1 offset to all bottom-row offsets.
		uint topRight = parallelScan2_shared[2 * WARP_SIZE - 1]<< 16;
		x0 += topRight;
		x1 += topRight;

		x0 -= sum0;
		x1 -= sum1;

		reduction2_shared[tid] = x0;
		reduction2_shared[WARP_SIZE + tid] = x1;
	} else if(numTransBuckets && 1 == warp) 
		// Expand the transaction list in parallel with the single-warp scan.
		ExpandScatterList(lane, numTransBuckets, compressed, uncompressed,
			debug_global);
		
	__syncthreads();

	predInc = scan[0];

	uint2 sortOffsets;
	scan = reduction2_shared + tid / STREAM_LENGTH_2;
	sortOffsets.x = scan[0];
	sortOffsets.y = scan[WARP_SIZE];

	// sortOffsets.x holds buckets 0 and 2
	// sortOffsets.y holds buckets 1 and 3
	// Expand predInc into shorts and add into sortOffsets.
	sortOffsets.x += prmt(predInc, 0, 0x4240);
	sortOffsets.y += prmt(predInc, 0, 0x4341);

	return sortOffsets;
}

////////////////////////////////////////////////////////////////////////////////


DEVICE void SortScatter2_8(uint2 scanOffsets, uint2 bucketsPacked, 
	uint2 localOffsets, Values fusedKeys, uint scatter[4], uint tid) {

	// scanOffsets holds scatter offsets within the warp for all 4 buckets.
	// These
	// must be added to globalOffsets[warp].
	// a holds offsets for buckets 0 and 2
	// b holds offsets for buckets 1 and 3
	uint a = scanOffsets.x;
	uint b = scanOffsets.y;

	// bucketsPacked hold a nibble for each bucket (from 0 to 3). We want to
	// convert each bucket code to the gather pair:
	// 0 -> 0x10 (lo short a)
	// 1 -> 0x54 (lo short b)
	// 2 -> 0x32 (hi short a)
	// 3 -> 0x76 (hi short b)
	// Gather the block offsets and add the warp offsets.
	uint codes0 = prmt(0x76325410, 0, bucketsPacked.x);
	scatter[0] = prmt(a, b, codes0) + ExpandUint8Low(localOffsets.x);
	scatter[1] = prmt(a, b, codes0>> 16) + ExpandUint8High(localOffsets.x);

	// Repeat the above instructions for values 4-7.
	uint codes1 = prmt(0x76325410, 0, bucketsPacked.y);
	scatter[2] = prmt(a, b, codes1) + ExpandUint8Low(localOffsets.y);
	scatter[3] = prmt(a, b, codes1>> 16) + ExpandUint8High(localOffsets.y);
}
