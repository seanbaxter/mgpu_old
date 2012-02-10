#pragma once

#include "common.cu"

// Runs a scan over the per-block digit counts.

template<int NumThreads, int NumBits>
DEVICE2 uint StridedMultiScan(uint tid, uint x, volatile uint* shared,
	volatile uint* totals_shared) {

	const int NumWarps = NumThreads / WARP_SIZE;
	const int NumDigits = 1<< NumBits;

	uint lane = (WARP_SIZE - 1) & tid;
	uint lane2 = (NumDigits - 1) & tid;
	uint warp = tid / WARP_SIZE;

	if(NumDigits < WARP_SIZE) {

		// Run an inclusive scan over repeated digits in the same warp.
		uint count = x;
		shared[tid] = x;

		#pragma unroll
		for(int i = 0; i < 5 - NumDigits; ++i) {
			int offset = NumDigits<< i;
			if(lane >= offset) x += shared[tid - offset];
			shared[tid - offset] = x;
		}
		uint laneExc = x - count;

		// Put the last digit in each warp (the warp total) into shared mem.
		__syncthreads();

		if((int)lane >= WARP_SIZE - NumDigits)
			shared[(WARP_SIZE + 1) * lane2 + warp] = x;
		__syncthreads();

		if(warp < NumDigits) {
			// Run NumDigits simultaneous parallel scans to sum up each digit.
			volatile uint* warpShared = shared + (WARP_SIZE + 1) * warp;

			x = warpShared[lane];
			uint count = x;

			#pragma unroll
			for(int i = 0; i < LOG_WARP_SIZE; ++i) {
				int offset = 1<< i;
				if(lane >= offset) x += warpShared[lane - offset];
				warpShared[lane] = x;
			}

			// Store the digit totals to shared mem.
			if(WARP_SIZE - 1 == lane)
				totals_shared[warp] = x;

			// Subtract count and put back in shared memory.
			warpShared[lane] = x - count;
		}
		__syncthreads();

		// Run the exclusive scan of digit totals.
		if(tid < NumDigits) {
			x = totals_shared[tid];
			uint count = x;

			#pragma unroll
			for(int i = 0; i < NumBits; ++i) {
				int offset = 1<< i;
				if(tid >= offset) x += totals_shared[tid - offset];
				if(i == NumBits - 1) x -= count;
				totals_shared[tid] = x;
			}
		}
		__syncthreads();

		// Add the three scanned values together for an exclusive offset for 
		// this lane.

		uint totalExc = shared[NumWarps * (WARP_SIZE + 1) + lane2];
		uint warpExc = shared[lane2 * (WARP_SIZE + 1) + warp];
		return totalExc + warpExc + laneExc;
	} else {
		// There are 32, 64, or 128 digits. Run a simple sequential scan.
		shared[tid] = x;
		__syncthreads();

		// Runs a scan with 1, 2, or 4 warps. Probably slower than parallel scan
		// but much easier to follow.
		if(tid < NumDigits) {
			const int NumDuplicates = NumThreads / NumDigits;
			uint x = 0;
			#pragma unroll
			for(int i = 0; i < NumDuplicates; ++i) {
				uint y = shared[i * NumDigits + tid];
				shared[i * NumDigits + tid] = x;
				x += y;
			}

			// Store the totals at the end of shared.
			totals_shared[tid] = x;
		}
		__syncthreads();

		IntraWarpParallelScan<NumBits>(tid, totals_shared, false);
		__syncthreads();

		uint exc = totals_shared[lane2] + shared[tid];
		return exc;
	}
}


template<int NumThreads, int NumBits>
DEVICE2 void SortHist(uint* blockTotals_global, uint numBlocks, 
	uint* totalsScan_global) {
	
	const int NumDigits = 1<< NumBits;
	const int NumColumns = NumThreads / NumDigits;

	__shared__ uint counts_shared[2 * NumThreads];
	__shared__ uint totals_shared[NumDigits];
	
	uint tid = threadIdx.x;
	uint lane2 = (NumDigits - 1) & tid;


	// Figure out which interval of the block counts to assign to each column.
	uint col = tid / NumDigits;
	uint quot = numBlocks / NumColumns;
	uint rem = (NumColumns - 1) & numBlocks;

	int2 range = ComputeTaskRange(col, quot, rem);
	
	uint start = NumDigits * range.x + lane2;
	uint end = NumDigits * range.y;
	uint stride = NumDigits;


	////////////////////////////////////////////////////////////////////////////
	// Upsweep pass. Divide the blocks up over the warps. We want warp 0 to 
	// process the first section of digit counts, warp 1 process the second 
	// section, etc.

	uint laneCount = 0;
	for(int i = start; i < end; i += stride)
		laneCount += blockTotals_global[i];

	// Run a strided multiscan to situate each lane within the global scatter
	// order.
	uint laneExc = StridedMultiScan<NumThreads, NumBits>(tid, laneCount, 
		counts_shared, totals_shared);

	if(totalsScan_global && (tid < NumDigits))
		totalsScan_global[tid] = totals_shared[tid];


	// Iterate over the block totals once again, adding and inc'ing laneExc.
	for(int i = start; i < end; i += stride) {
		uint blockCount = blockTotals_global[i];
		blockTotals_global[i] = laneExc;
		laneExc += blockCount;
	}
}


#define GEN_SORTHIST_FUNC(Name, NumThreads, NumBits, BlocksPerSM)			\
																			\
extern "C" void __global__ Name(uint* blockTotals_global, uint numBlocks,	\
	uint* totalsScan_global) {												\
	SortHist<NumThreads, NumBits>(blockTotals_global, numBlocks,			\
		totalsScan_global);													\
}

GEN_SORTHIST_FUNC(SortHist_1, 1024, 1, 1)
GEN_SORTHIST_FUNC(SortHist_2, 1024, 2, 1)
GEN_SORTHIST_FUNC(SortHist_3, 1024, 3, 1)
GEN_SORTHIST_FUNC(SortHist_4, 1024, 4, 1)
GEN_SORTHIST_FUNC(SortHist_5, 1024, 5, 1)
GEN_SORTHIST_FUNC(SortHist_6, 1024, 6, 1)
GEN_SORTHIST_FUNC(SortHist_7, 1024, 7, 1)
