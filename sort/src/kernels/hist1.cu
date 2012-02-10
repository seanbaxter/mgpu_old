////////////////////////////////////////////////////////////////////////////////
// PHASE ONE
// Build a running sum of all the bucket counts per column. This is a vertical 
// sum. Then perform a horizontal add from left-to-right to get the total number
// of buckets for this histogram block.

// Input:
// bucketCount_global is the output of the count kernel.
//		Bucket counts for each sort block packed into 16bit uints.
// rangePairs_global is the start and end interval for each warp in this pass.
//		These values always refers to warp's worth of data, so multiply by 32.
// Output:
// countScan_global is the total count for each bucket. Output for each
//		histogram block is NUM_BUCKETS, totalling NumSMs * NUM_BUCKETS. These 
//		values are scanned and modified in-place by the phase 2 histogram 
//		kernel.
// columnScan_global is the scan of bucket counts for totals within each warp of
//		this block. This is required as it gives each warp in the phase 3 
//		histogram kernel a starting scatter offset for each bucket. This data
//		is not read by the phase 2 kernel. However, the countScan_globals are 
//		modified by the phase 2 kernel and added to columnScan_global in phase
//		3.
//		
// The 16-bit totals from the count kernel are expanded to 32 bits in this
// kernel.

#pragma once

#include "common.cu"
#include "params.cu"

template<int NumThreads, int NumBits>
DEVICE2 void HistogramFunc1(const uint* bucketCount_global, int rangeQuot,
	int rangeRem, int segSize, int count, uint* countScan_global,
	uint* columnScan_global) {

	const int NumWarps = NumThreads / WARP_SIZE;
	const int NumDigits = 1<< NumBits;
	const int NumChannels = NumDigits / 2;
	const int WarpStride = MAX(NumChannels, WARP_SIZE);
	const int BlocksPerWarp = WarpStride / NumChannels;

	__shared__ volatile uint hist_shared[4 * NumThreads];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	////////////////////////////////////////////////////////////////////////////
	// Iterate over all the block digit counts and accumulate.

	// uint2 range = rangePairs_global[NUM_WARPS * block + warp];
	int2 range = ComputeTaskRange(NumWarps * block + warp, rangeQuot, rangeRem,
		segSize, count);

	// Running sum only for this thread. For NumBits = 7 (NumDigits = 128), 
	// we have 64 elements. The first warp of values holds (0, 64) - (31, 95).
	// The second warp of values holds (32, 96) - (63, 127). These are unpacked
	// into 32-bit registers.

	// For NumBits <= 6, only count0 and count1 are used. count0 holds the least
	// significant count from each loaded value, and count1 holds the most 
	// significant count.

	// For 7 == NumBits:
	// count0: 0-31, count1: 64-95, count2: 32-63, count3: 96-127.
	uint count0 = 0;
	uint count1 = 0;
	uint count2 = 0;
	uint count3 = 0;

	uint current = range.x;
	while(current < range.y) {
		uint packed = bucketCount_global[WarpStride * current + lane];

		// The most sig bit of the lo short is a flag indicating sort detect.
		// Clear this bit in hist1, and read it in hist3.
		count0 += 0x00007fff & packed;
		count1 += packed>> 16;

		if(7 == NumBits) {
			uint packed2 = bucketCount_global[
				WarpStride * current + WARP_SIZE + lane];
			count2 += 0xffff & packed2;
			count3 += packed>> 16;
		}
		++current;
	}


	////////////////////////////////////////////////////////////////////////////
	// Store the counts in shared memory and run a simple reduction over 
	// duplicate digit counts within the same warp.

	hist_shared[tid] = count0;
	hist_shared[NumThreads + tid] = count1;
	if(7 == NumBits) {
		hist_shared[2 * NumThreads + tid] = count2;
		hist_shared[3 * NumThreads + tid] = count3;
	}

	if(lane < NumChannels) {
		#pragma unroll
		for(int i = 1; i < BlocksPerWarp; ++i) {
			count0 += hist_shared[tid + i * NumChannels];
			count1 += hist_shared[NumThreads + tid + i * NumChannels];
		}
	}
	__syncthreads();

	volatile uint* warp_shared = hist_shared + NumDigits * warp;
	if(NumBits <= 6) {
		if(lane < NumChannels) {
			warp_shared[lane] = count0;
			warp_shared[NumChannels + lane] = count1;
		}
	} else if(7 == NumBits) {
		warp_shared[lane] = count0;
		warp_shared[32 + lane] = count2;
		warp_shared[64 + lane] = count1;
		warp_shared[96 + lane] = count3;
	}
	__syncthreads();

	
	////////////////////////////////////////////////////////////////////////////
	// Perform inter-warp reduction for bucket totals for each warp.

	if(tid < NumDigits) {
		uint x = 0;

		uint index = NumDigits * NumWarps * block;
		#pragma unroll
		for(int i = 0; i < NumWarps; ++i) {
			// Output the scan values.
			columnScan_global[index + tid + i * NumDigits] = x;
			x += hist_shared[tid + i * NumDigits];
		}

		// Output the digit totals.
		countScan_global[block * NumDigits + tid] = x;
	}
}


#define GEN_HIST1_FUNC(Name, NumThreads, NumBits)							\
																			\
extern "C" __global__ void Name(const uint* bucketCount_global,				\
	int rangeQuot, int rangeRem, int segSize, int count,					\
	uint* countScan_global, uint* columnScan_global) {						\
																			\
	HistogramFunc1<NumThreads, NumBits>(bucketCount_global, rangeQuot,		\
		rangeRem, segSize, count, countScan_global, columnScan_global);		\
}
