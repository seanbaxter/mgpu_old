////////////////////////////////////////////////////////////////////////////////
// REDUCTION PASS kernel. Used by both segscanflags and segscankeys.

extern "C" __global__ __launch_bounds__(REDUCTION_NUM_THREADS, 1)
void SegScanReduction(const uint* headFlags_global, uint* blockLast_global,
	uint numBlocks) {

	const int NumWarps = REDUCTION_NUM_THREADS / WARP_SIZE;
	const int LogNumWarps = LOG_BASE_2(NumWarps);

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	// Load the head flag and last segment counts for each thread. These map
	// to blocks in the upsweep/downsweep passes.
	uint flag = 0;
	uint x = 0;
	if(tid < numBlocks) {
		flag = headFlags_global[tid];
		x = blockLast_global[tid];
	}

	// Get the start flags for each thread in the warp.
	uint flags = __ballot(flag);

	// Mask out the bits at or above the current lane.
	uint mask = bfi(0, 0xffffffff, 0, lane);
	uint flagsMasked = flags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	int preceding = 31 - __clz(flagsMasked);
	uint distance = lane - preceding;

	__shared__ volatile uint shared[NumWarps * (WARP_SIZE + 1)];
	__shared__ volatile uint blockShared[2 * NumWarps];
	volatile uint* warpShared = shared + warp * (WARP_SIZE + 1) + 1;
	volatile uint* threadShared = warpShared + lane;

	// Run an inclusive scan for each warp. This does not require any special 
	// treatment of segment edges, as we have only one value per thread.
	threadShared[-1] = 0;
	threadShared[0] = x;
	uint sum = x;
	uint first = warpShared[preceding];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		if(distance > offset)
			sum += threadShared[-offset];
		threadShared[0] = sum; 
	}
	sum += first;

	uint last = flag ? x : sum;

	// sum now holds the inclusive scan for the part of the segment within the
	// warp. Run a multiscan by having each warp store its flags value to
	// shared memory.
	if(WARP_SIZE - 1 == lane) {
		blockShared[warp] = flags;
		blockShared[NumWarps + warp] = last;
	}

	__syncthreads();
	if(tid < NumWarps) {

		// Load the inclusive sums for the last value in each warp and the head
		// flags for each warp.
		uint flag = blockShared[tid];
		uint x = blockShared[NumWarps + tid];
		uint flags = __ballot(flag) & mask;

		int preceding = 31 - __clz(flags);
		uint distance = tid - preceding;

		volatile uint* s = blockShared + NumWarps + tid;
		s[-NumWarps] = 0;

		uint sum = x;
		uint first = blockShared[NumWarps + preceding];

		#pragma unroll
		for(int i = 0; i < LogNumWarps; ++i) {
			int offset = 1<< i;
			if(distance > offset) sum += s[-offset];
			s[0] = sum;
		}

		// Add preceding and subtract x to get an exclusive sum.
		sum += first - x;

		blockShared[tid] = sum;
	}

	__syncthreads();

	uint blockScan = blockShared[warp];

	// Add blockScan if the warp doesn't hasn't encountered a head flag yet.
	if(!flagsMasked) sum += blockScan;
	sum -= x;

	if(tid < numBlocks)
		blockLast_global[tid] = sum;
}
