#include "histcommon.cu"

template<typename T>
DEVICE2 void SelectCountItem(const T* source_global, uint lane, int& offset,
	int end, uint shift, uint bits, int loopCount, volatile uint* warpShared,
	uint threadTotals[2], bool check, int& dumpTime, bool forceExpand) {

	#pragma unroll
	for(int i = 0; i < loopCount; ++i) {
		int source = offset + i * WARP_SIZE + lane;
		if(!check || (source < end)) {
			T x = source_global[source];
			uint digit = bfe(ConvertToUint(x), shift, bits);
			IncCounter(warpShared + lane, digit);
		}
	}
	dumpTime += loopCount;
	offset += loopCount * WARP_SIZE;

	if(forceExpand || (dumpTime >= (256 - loopCount))) {
		ExpandCounters(lane, warpShared, threadTotals);
		dumpTime = 0;
	}
}

template<typename T>
DEVICE2 void SelectCount(const T* source_global, uint* hist_global, 
	const int2* range_global, uint shift, uint bits) {

	// Process four values at a time in the inner loop.
	const int InnerLoop = 4;

	__shared__ volatile uint counts_shared[NUM_THREADS * 16];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile uint* warpShared = counts_shared + 16 * WARP_SIZE * warp;
	ClearCounters(warpShared + lane);

	// Each warp holds two unpacked totals.
	uint threadTotals[2] = { 0, 0 };
	int dumpTime = 0;

	uint end = ROUND_DOWN(range.y, WARP_SIZE * InnerLoop);

	// Use loop unrolling for the parts that aren't at the end of the array.
	// This reduces the logic for bounds checking.
	while(range.x < end)
		SelectCountItem(source_global, lane, range.x, end, shift, bits, 
			InnerLoop, warpShared, threadTotals, false, dumpTime, false);

	// Process the end of the array. For an expansion of the counters.
	SelectCountItem(source_global, lane, range.x, range.y, shift, bits, 
		InnerLoop, warpShared, threadTotals, true, dumpTime, true);

	hist_global[64 * gid + 2 * lane] = threadTotals[0];
	hist_global[64 * gid + 2 * lane + 1] = threadTotals[1];
}


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SelectCountUint(const uint* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	SelectCount(source_global, hist_global, range_global, shift, bits);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SelectCountInt(const int* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	SelectCount(source_global, hist_global, range_global, shift, bits);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SelectCountFloat(const float* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	SelectCount(source_global, hist_global, range_global, shift, bits);
}

