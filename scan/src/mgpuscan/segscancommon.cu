#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 3

#define BLOCKS_PER_SM 2

#define VALUES_PER_THREAD 16
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)


////////////////////////////////////////////////////////////////////////////////
// Reduction function for upsweep pass. This performs addition/prefix sum for
// code 0 and max for code 1.

DEVICE int Reduce(uint tid, int x, int code) {

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	int init = code ? -1 : 0;

	const int ScanStride = WARP_SIZE + WARP_SIZE / 2 + 1;
	const int ScanSize = NUM_WARPS * ScanStride;
	__shared__ volatile int reduction_shared[ScanSize];
	__shared__ volatile int totals_shared[NUM_WARPS + NUM_WARPS / 2];

	volatile int* s = reduction_shared + ScanStride * warp + lane + 
		WARP_SIZE / 2;
	s[-16] = init;
	s[0] = x;

	// Run intra-warp max reduction.
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(0 == code) x += s[-offset];
		else if(1 == code) x = max(x, s[-offset]);
		s[0] = x;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		x = reduction_shared[ScanStride * tid + WARP_SIZE / 2 +
			WARP_SIZE - 1];

		totals_shared[tid] = init;
		volatile int* s = totals_shared + NUM_WARPS / 2 + tid;
		s[0] = x;
		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(0 == code) x += s[-offset];
			else if(1 == code) x = max(x, s[-offset]);
			s[0] = x;
		}

		if(NUM_WARPS - 1 == tid) totals_shared[0] = x;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	return totals_shared[0];
}


////////////////////////////////////////////////////////////////////////////////
// REDUCTION PASS kernel. Used by both segscanflags and segscankeys.

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void SegScanReduction(const uint* headFlags_global, uint* blockLast_global,
	uint numBlocks) {

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

	__shared__ volatile uint shared[NUM_WARPS * (WARP_SIZE + 1)];
	__shared__ volatile uint blockShared[2 * NUM_WARPS];
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
		blockShared[NUM_WARPS + warp] = last;
	}

	__syncthreads();
	if(tid < NUM_WARPS) {

		// Load the inclusive sums for the last value in each warp and the head
		// flags for each warp.
		uint flag = blockShared[tid];
		uint x = blockShared[NUM_WARPS + tid];
		uint flags = __ballot(flag) & mask;

		int preceding = 31 - __clz(flags);
		uint distance = tid - preceding;

		volatile uint* s = blockShared + NUM_WARPS + tid;
		s[-NUM_WARPS] = 0;

		uint sum = x;
		uint first = blockShared[NUM_WARPS + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
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


////////////////////////////////////////////////////////////////////////////////
// INTER-WARP REDUCTION 
// Calculate the length of the last segment in the last lane in each warp. Also
// store the block offset to shared memory for the next pass.

DEVICE uint BlockScan(uint tid, uint warp, uint lane, uint last,
	uint warpFlags, uint mask, volatile uint* blockOffset_shared) {

	__shared__ volatile uint blockShared[3 * NUM_WARPS];
	if(WARP_SIZE - 1 == lane) {
		blockShared[NUM_WARPS + warp] = last;
		blockShared[2 * NUM_WARPS + warp] = warpFlags;
	}
	__syncthreads();

	if(tid < NUM_WARPS) {
		// Pull out the sum and flags for each warp.
		volatile uint* s = blockShared + NUM_WARPS + tid;
		uint warpLast = s[0];
		uint flag = s[NUM_WARPS];
		s[-NUM_WARPS] = 0;

		uint blockFlags = __ballot(flag);

		// Mask out the bits at or above the current warp.
		blockFlags &= mask;

		// Find the distance from the current warp to the warp at the start of 
		// this segment.
		int preceding = 31 - __clz(blockFlags);
		uint distance = tid - preceding;


		// INTER-WARP reduction
		uint warpSum = warpLast;
		uint warpFirst = blockShared[NUM_WARPS + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			uint offset = 1<< i;
			if(distance > offset) warpSum += s[-offset];
			if(i < LOG_NUM_WARPS - 1) s[0] = warpSum;
		}
		// Subtract warpLast to make exclusive and add first to grab the
		// fragment sum of the preceding warp.
		warpSum += warpFirst - warpLast;

		// Store warpSum back into shared memory. This is added to all the
		// lane sums and those are added into all the threads in the first 
		// segment of each lane.
		blockShared[tid] = warpSum;

		// Set the block offset for the next brick of data.
		if(NUM_WARPS - 1 == tid) {
			if(!flag) warpLast += warpSum;
			*blockOffset_shared = warpLast;
		}
	}
	__syncthreads();

	return blockShared[warp];
}


////////////////////////////////////////////////////////////////////////////////
// Segmented scan downsweep logic. Abstracts away loading of values and head 
// flags.

DEVICE void SegScanDownsweep(uint tid, uint lane, uint warp, 
	uint x[VALUES_PER_THREAD], const uint flags[VALUES_PER_THREAD],
	volatile uint* warpShared, volatile uint* threadShared, bool inclusive, 
	volatile uint* blockOffset_shared) {

	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP PASS
	// Add sum to all the values in the continuing segment (that is, before the
	// first start flag) in this thread.

	uint blockOffset = 0;
	if(!tid) blockOffset = *blockOffset_shared;
	uint last = blockOffset;

	// Compute the exclusive scan into scan. These values are then added to the
	// final thread offsets after the inter-warp multiscan pattern.
	uint hasHeadFlag = 0;

	if(inclusive) {
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			if(flags[i]) last = 0;
			hasHeadFlag |= flags[i];

			x[i] += last;
			last = x[i];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			if(flags[i]) last = 0;
			if(flags[i]) hasHeadFlag |= flags[i];
			uint val = x[i];

			x[i] = last;
			last += val;
		}
	}

	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP SEGMENT PASS
	// Run a ballot and clz to find the lane containing the start value for the
	// segment that begins this thread.

	uint warpFlags = __ballot(hasHeadFlag);

	// Mask out the bits at or above the current thread.
	uint mask = bfi(0, 0xffffffff, 0, lane);
	uint warpFlagsMask = warpFlags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	int preceding = 31 - __clz(warpFlagsMask);
	uint distance = lane - preceding;


	////////////////////////////////////////////////////////////////////////////
	// REDUCTION PASS
	// Run a prefix sum scan over last to compute for each lane the sum of all
	// values in the segmented preceding the current lane, up to that point.
	// This is added back into the thread-local exclusive scan for the continued
	// segment in each thread.

	volatile uint* shifted = threadShared + 1;
	shifted[-1] = 0;
	shifted[0] = last;
	uint sum = last;
	uint first = warpShared[1 + preceding];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(distance > offset) sum += shifted[-offset];
		if(i < LOG_WARP_SIZE - 1) shifted[0] = sum;
	}
	// Subtract last to make exclusive and add first to grab the fragment
	// sum of the preceding thread.
	sum += first - last;

	// Call BlockScan for inter-warp scan on the reductions of the last
	// segment in each warp.
	uint lastSegLength = last;
	if(!hasHeadFlag) lastSegLength += sum;

	uint blockScan = BlockScan(tid, warp, lane, lastSegLength, warpFlags, 
		mask, blockOffset_shared);
	if(!warpFlagsMask) sum += blockScan;

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		if(flags[i]) sum = 0;
		x[i] += sum;
	}
}


#include "segscanflags.cu"
#include "segscankeys.cu"

#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES

