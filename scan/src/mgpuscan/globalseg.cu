
#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 3

#define BLOCKS_PER_SM 4

#define VALUES_PER_THREAD 8
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)

// Use a 33-slot stride for shared mem transpose.
#define WARP_STRIDE (WARP_SIZE + 1)

/*
#define TRANSPOSE_SHARED_SIZE (NUM_WARPS * WARP_STRIDE * VALUES_PER_THREAD)
#define BLOCK_SHARED_SIZE (3 * NUM_WARPS)

volatile uint seg_shared[TRANSPOSE_SHARED_SIZE];
volatile uint block_shared[BLOCK_SHARED_SIZE];
*/

////////////////////////////////////////////////////////////////////////////////
// UPSWEEP PASS. Find the sum of all values in the last segment in each block.
// When the first head flag in the block is encountered, write out the sum to 
// that point and return. We only need to reduce the last segment to feed sums
// up to the reduction pass.

DEVICE int Reduce(uint tid, int x, int code, int init) {

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

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
		volatile int* s2 = totals_shared + NUM_WARPS / 2 + tid;
		s2[0] = x;
		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(0 == code) x += s[-offset];
			else if(1 == code) x = max(x, s[-offset]);
			s2[0] = x;
		}

		if(NUM_WARPS - 1 == tid) totals_shared[0] = x;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	return totals_shared[0];
}


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepFlag(const uint* valuesIn_global, uint* valuesOut_global,
	int* headFlagPos_global, const int2* rangePairs_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	int2 range = rangePairs_global[block];
	
	// Round the end iterator down to a multiple of NUM_VALUES.
	int current = ~NUM_VALUES & (range.y - 1);

	// range.x was set by the host to be aligned. We need only check the ranges
	// for the last brick in each block (that is, the first one processed).
	bool checkRange = (block == gridDim.x - 1);
	
	uint threadSum = 0;
	int segmentStart = -1;

	while(current >= range.x) {

		uint packed[VALUES_PER_THREAD];
		if(checkRange) {
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				int index = current + tid + i * NUM_THREADS;
				packed[i] = 0;
				if(index < range.y)
					packed[i] = valuesIn_global[index];
			}
		} else {
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) 
				packed[i] = valuesIn_global[current + tid + i * NUM_THREADS];
		}

		// Find the index of the latest value loaded with a head flag set.
		int lastHeadFlagPos = -1;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint flag = 0x80000000 & packed[i];
			if(flag) lastHeadFlagPos = i;
		}
		if(-1 != lastHeadFlagPos)
			lastHeadFlagPos = tid + lastHeadFlagPos * NUM_THREADS;

		segmentStart = Reduce(tid, lastHeadFlagPos, 1, -1);

		// Make a second pass and sum all the values that appear at or after
		// segmentStart.

		// Add if tid + i * NUM_THREADS >= segmentStart.
		// Subtract tid from both sides to simplify expression.
		int cmp = segmentStart - tid;
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint value = 0x7fffffff & packed[i];
			if(i * NUM_THREADS >= cmp)
				threadSum += value;
		}
		if(-1 != segmentStart) break;

		current -= NUM_VALUES;
		checkRange = false;
	}

	// We've either hit the head flag or run out of values. Do a horizontal sum
	// of the thread values and store to global memory.
	uint total = (uint)Reduce(tid, (int)threadSum, 0, 0);

	if(0 == tid) {
		valuesOut_global[block] = total;
		if(-1 != segmentStart) segmentStart += current;
		headFlagPos_global[block] = segmentStart;
	}
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepKeys(const uint* valuesIn_global, const uint* keysIn_global,
	uint* valuesOut_global, const int2* rangePairs_global) {


}


////////////////////////////////////////////////////////////////////////////////
// REDUCTION PASS. 

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void SegScanReduction(const uint* headFlags_global, uint* sums_global,
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
		x = sums_global[tid];
	}

	// Get the start flags for each thread in the warp.
	uint flags = __ballot(flag);

	// Mask out the bits above the current lane.
	uint flagsMasked = flags & bfi(0, 0xffffffff, 0, lane + 1);

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	uint distance = __clz(flags) + lane - 31;






	



}
const uint* valuesIn_global, uint* valuesOut_global,
	int* headFlagPos_global,

////////////////////////////////////////////////////////////////////////////////
// DOWNSWEEP PASS. Add the 

