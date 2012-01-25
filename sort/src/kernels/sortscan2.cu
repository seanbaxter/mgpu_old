
DEVICE uint2 MultiScan2(uint tid, uint predInc, uint numThreads, uint packed[4],
	uint* scratch_shared, uint* debug_global) {

	const int NumValues = VALUES_PER_THREAD * numThreads;
	const int NumWarps = numThreads / WARP_SIZE;

	// Allocate 1 int for each thread to store its digit count. These are
	// strided for fast parallel scan access, so consume 33 values per warp.
	const int ScanSize = numThreads + NumWarps;
	volatile uint* predInc_shared = (volatile uint*)scratch_shared;

	// Each stream has StreamLen elements. 
	const int StreamLen = NumWarps;
	const int StreamsPerWarp = WARP_SIZE / NumWarps;

	// Store the stream totals and do a parallel scan. We need to scan 64
	// elements, as each stream total now takes a full 16bits.
	const int ParallelScanSize = 2 * WARP_SIZE + 16;
	volatile uint* parallelScan_shared = predInc_shared + ScanSize + 16;

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	// Store the byte-packed values here.
	volatile uint* scan = predInc_sared + tid + tid / WARP_SIZE;
	scan[0] = predInc;
	__syncthreads();

	// Perform sequential scan over the byte-packed counts.
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
		uint topRight = parallelScan_shared[2 * WARP_SIZE - 1]<< 16;
		x0 += topRight;
		x1 += topRight;

		x0 -= sum0;
		x1 -= sum1;

		parallelScan_shared[tid] = x0;
		parallelScan_shared[WARP_SIZE + tid] = x1;
	}		
	__syncthreads();

	predInc = scan[0];

	uint2 sortOffsets;
	scan = parallelScan_shared + tid / StreamLen;
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
// Combine the sortOffsets (computed above) with the digits and local offsets
// for packed scatter offsets.

DEVICE void SortScatter2_8(uint2 scanOffsets, uint2 bucketsPacked, 
	uint2 localOffsets, uint scatter[4]) {

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
