
////////////////////////////////////////////////////////////////////////////////
// PHASE TWO
// After all blocks synchronize after writing their total bucket counts, the
// first block performs an in-place inclusive scan of the bucket counts. This 
// is a straight-forward linear scan.

// Lots of __syncthreads but who cares? On all current hardware (16 SMs or
// fewer), it only takes one inner loop (64 buckets * 16 SM = 1024, the max 
// number of threads).

#pragma once

#include "common.cu"
#include "params.cu"



template<int NumThreads, int NumBits>
DEVICE2 void HistogramFunc2(uint numBlocks, uint* countScan_global,
	uint* digitTotals_global) {
	
	const int NumDigits = 1<< NumBits;

	__shared__ volatile uint counts_shared[NumDigits * MAX_DEVICE_SMS];
	__shared__ volatile uint digitTotals_shared[NumDigits];

	// Perform an in-place scan over countScan_global.
	uint tid = threadIdx.x;
	int numCounts = NumDigits * numBlocks;

	////////////////////////////////////////////////////////////////////////////
	// Read in all block totals (from hist 1) and scan by digit, then store
	// the digit totals to shared memory.

	// Read in all the counts from hist1 pass.
	for(int i = tid; i < numCounts; i += NumThreads)
		counts_shared[i] = countScan_global[i];
	__syncthreads();

	// Run a sequential scan over all the blocks. This isn't optimal, but it's 
	// a trivial amount of data to process and we should keep this simple.
	if(tid < NumDigits) {	
		uint x = 0;
		for(int i = tid; i < numCounts; i += NumDigits) {
			uint y = counts_shared[i];
			counts_shared[i] = x;
			x += y;
		}
		digitTotals_shared[tid] = x;
		if(digitTotals_global)
			digitTotals_global[tid] = x;
	}
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Parallel scan the global digit totals. Add the exclusive bucket sums to
	// the block offsets and write back to countScan_global.

	IntraWarpParallelScan<NumBits>(tid, digitTotals_shared, false);
	__syncthreads();

	for(int i = tid; i < numCounts; i += NumThreads) {
		uint digitScan = digitTotals_shared[(NumDigits - 1) & i];
		uint bucketOffset = counts_shared[i];
		countScan_global[i] = digitScan + bucketOffset;
	}
}


#define GEN_HIST2_FUNC(Name, NumThreads, NumBits)							\
																			\
extern "C" __global__ void Name(uint numBlocks, uint* countScan_global,		\
	uint* digitTotals_global) {												\
																			\
	HistogramFunc2<NumThreads, NumBits>(numBlocks, countScan_global,		\
		digitTotals_global);												\
}
