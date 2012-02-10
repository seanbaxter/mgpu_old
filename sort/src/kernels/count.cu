#pragma once

#include "countcommon.cu"

////////////////////////////////////////////////////////////////////////////////
// CountFunc

// Have each warp process a histogram for one block. If the block has
// NUM_VALUES, each thread process NUM_VALUES / WARP_SIZE. Eg if NUM_VALUES is
// 2048, each thread process 64 values. This operation is safe for NUM_VALUES up
// to 4096.
template<int NumBits, int NumThreads, int InnerLoop, int Mode>
DEVICE2 void CountFunc(const uint* keys_global, uint bit, uint numElements, 
	uint vt, uint* counts_global) {

	const int NumBuckets = 1<< NumBits;
	const int NumCounters = DIV_UP(NumBuckets, 4);
	const int NumChannels = NumBuckets / 2;
	
	const int NumWarps = NumThreads / WARP_SIZE;

	const int WarpMem = WARP_SIZE * NumCounters;

	__shared__ volatile uint blockCounters_shared[NumCounters * NumThreads];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint warpStart = (block * NumWarps + warp) * (WARP_SIZE * vt);

	volatile uint* warpCounters_shared = blockCounters_shared + warp * WarpMem;
	volatile uint* counters_shared = warpCounters_shared + lane;

	// Define the counters so we can pass them to IncBucketCounter. They don't
	// actually get used unless NUM_BITS <= 3 however.
	uint counter0 = 0;
	uint counter1 = 0;

	// clear all the counters
	#pragma unroll
	for(int i = 0; i < NumCounters; ++i)
		counters_shared[WARP_SIZE * i] = 0;

	uint values[InnerLoop];
	if(warpStart < numElements) {
		// Unroll to read 8 values at a time. We can also read from uint4!
		const uint* warpData = keys_global + warpStart + lane;
		uint end = vt / InnerLoop;

		// Consume InnerLoop values at a time.
		for(int i = 0; i < end; ++i) {
			
			// Load InnerLoop values from global memory.
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j)
				values[j] = warpData[j * WARP_SIZE];
			warpData += InnerLoop * WARP_SIZE;

			// Extract the digit and increment the bucket counter.
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j) {
				uint digit = bfe(values[j], bit, NumBits);
				IncBucketCounter(digit, counters_shared, counter0, counter1,
					NumBits);
			}
		}

		// Write the counters to shared memory if they were stored in register.
		if(NumBits <= 3)
			counters_shared[0] = counter0;
		if(3 == NumBits)
			counters_shared[WARP_SIZE] = counter1;
	}

	uint2 countPair = GatherSumsReduce<NumBits>(warpCounters_shared, lane, Mode,
		blockCounters_shared);
	__syncthreads();

	volatile uint* digitTotals = blockCounters_shared + NumChannels * warp;
	if(NumBits <= 6) {
		if(lane < NumChannels)
			digitTotals[lane] = countPair.x;		
	} else if(7 == NumBits) {
		digitTotals[2 * lane + 0] = countPair.x;
		digitTotals[2 * lane + 1] = countPair.y;
	}
	__syncthreads();

	// Store the counts to global memory.
	uint offset = NumWarps * NumChannels * block;
	for(int i = tid; i < NumChannels * NumWarps; i += NumThreads)
		counts_global[offset + i] = blockCounters_shared[i];
}

#define GEN_COUNT_FUNC(Name, NumThreads, NumBits, InnerLoop, Mode,			\
	BlocksPerSM)															\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global, uint bit, uint numElements, uint vt,		\
	uint* counts_global) {													\
																			\
	CountFunc<NumBits, NumThreads, InnerLoop, Mode>(						\
		keys_global, bit, numElements, vt, counts_global);					\
}


////////////////////////////////////////////////////////////////////////////////
// CountFuncLoop 

template<int NumBits, int NumThreads, int InnerLoop, int Mode>
DEVICE2 void CountFuncLoop(const uint* keys_global, uint bit, uint vt, 
	int taskQuot, int taskRem, uint* counts_global, uint* totals_global) {

	const int NumBuckets = 1<< NumBits;
	const int NumCounters = DIV_UP(NumBuckets, 4);
	const int NumChannels = NumBuckets / 2;
	
	const int NumWarps = NumThreads / WARP_SIZE;

	const int WarpMem = WARP_SIZE * NumCounters;

	__shared__ volatile uint blockCounters_shared[NumCounters * NumThreads];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;

	volatile uint* warpCounters_shared = blockCounters_shared + warp * WarpMem;
	volatile uint* counters_shared = warpCounters_shared + lane;

	int task = NumWarps * block + warp;
	int2 rangePair = ComputeTaskRange(task, taskQuot, taskRem);

	// Offset the keys and counts pointers.

	// Initialize the unpacked digit counters for each lane.
	uint laneCount0 = 0, laneCount1 = 0, laneCount2 = 0, laneCount3 = 0;
	
	int current = rangePair.x;
	uint end = vt / InnerLoop;
	while(current < rangePair.y) {
	
		const uint* keys_pass = keys_global + current * (WARP_SIZE * vt);
		uint* counts_pass = counts_global + current * NumChannels;

		const uint* warpData = keys_pass + lane;
			
		// Clear all the counters
		#pragma unroll
		for(int i = 0; i < NumCounters; ++i)
			counters_shared[WARP_SIZE * i] = 0;
		uint counter0 = 0;
		uint counter1 = 0;

		// Consume InnerLoop values at each iteration. This increases ILP and
		// decreases the amount of branching.
		for(int i = 0; i < end; ++i) {
			
			// TODO: try replacing with uint4 loads to decrease instruction 
			// count and increase ILP.
			uint values[InnerLoop];
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j)
				values[j] = warpData[j * WARP_SIZE];
			warpData += InnerLoop * WARP_SIZE;

			// Extract the digit and increment the bucket counter.
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j) {
				uint digit = bfe(values[j], bit, NumBits);
				IncBucketCounter(digit, counters_shared, counter0, counter1,
					NumBits);
			}
		}


		// Write the counters to shared memory if they were stored in register.
		if(NumBits <= 3)
			counters_shared[0] = counter0;
		if(3 == NumBits)
			counters_shared[WARP_SIZE] = counter1;

		uint2 countPair = GatherSumsReduce<NumBits>(warpCounters_shared, lane, 
			Mode, blockCounters_shared);

		// NOTE: exclusive scan countPair and even pre-subtract some terms to
		// reduce computation on 

		// Store the counts to global memory.
		if(NumBits <= 6) {
			if(lane < NumChannels) counts_pass[lane] = countPair.x;
		} else if(7 == NumBits) {
			((uint2*)counts_pass)[lane] = countPair;
		}

		laneCount0 += 0x0000ffff & countPair.x;
		laneCount1 += countPair.x>> 16;
		if(7 == NumBits) {
			laneCount2 += 0x0000ffff & countPair.y;
			laneCount3 += countPair.y>> 16;
		}

		++current;
	}

	// Store the digit totals to global memory. Each warp stores NumDigit
	// values.
	if(NumBits <= 6) {
		// Unpack and order by ascending digit count. 
		if(lane < NumChannels) {
			uint* totals = totals_global + NumBuckets * task;
			totals[lane] = laneCount0;
			totals[NumChannels + lane] = laneCount1;
		}
	} else if(7 == NumBits) {
		// lane counts 0 and 2 hold adjacent values (0 + lane and 1 + lane),
		// as do lane counts 1 and 3 (64 + lane and 65 + lane).
		
		uint2* totals = (uint2*)(totals_global + NumBuckets * task);
		totals[lane] = make_uint2(laneCount0, laneCount2);
		totals[WARP_SIZE + lane] = make_uint2(laneCount1, laneCount3);
	}
}

#define GEN_COUNT_LOOP(Name, NumThreads, NumBits, InnerLoop, Mode,			\
	BlocksPerSM)															\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global, uint bit, uint vt, int taskQuot,			\
	int taskRem, uint* counts_global, uint* totals_global) {				\
																			\
	CountFuncLoop<NumBits, NumThreads, InnerLoop, Mode>(					\
		keys_global, bit, vt, taskQuot, taskRem, counts_global,				\
		totals_global);														\
}

