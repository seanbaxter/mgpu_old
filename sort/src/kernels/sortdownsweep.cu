#pragma once

#include "common.cu"
#include "params.cu"

////////////////////////////////////////////////////////////////////////////////
// SortDownsweep

template<int NumBits, int NumThreads>
DEVICE2 void SortDownsweep(const uint* taskScan_global, int taskQuot,
	int taskRem, int numBlocks, int blockSize, 
	const uint* blockLocalScan_global, uint* blockGlobalScan_global) {

	const int NumDigits = 1<< NumBits;
	const int NumChannels = NumDigits / 2;
	const int ColumnWidth = MIN(NumDigits, WARP_SIZE);
	const int NumColumns = NumThreads / ColumnWidth;
	const int CountsPerThread = NumDigits / ColumnWidth;
	
	const int WarpMem = NumDigits + 1;

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	
	uint column = tid / ColumnWidth;  
	uint lane = (ColumnWidth - 1) & tid;
	uint task = NumColumns * block + column;
	int2 taskRange = ComputeTaskRange(task, taskQuot, taskRem, 1, numBlocks);

	__shared__ volatile uint shared[NumColumns * WarpMem];
	volatile uint* column_shared = shared + WarpMem * column;

	// Set the scaled number of values per sort block into the last shared mem
	// slot for this warp. This lets us derive digit counts using the difference
	// between adjacent scan offsets.
	if(!lane) column_shared[NumDigits] = 4 * blockSize;

	// Initialize the task offsets.
	uint taskOffsets[4];
	if(taskRange.x < taskRange.y) {
		taskOffsets[0] = taskScan_global[NumDigits * task + lane];
		if(NumBits >= 6)
			taskOffsets[1] = taskScan_global[NumDigits * task + 32 + lane];
		if(NumBits >= 7) {
			taskOffsets[2] = taskScan_global[NumDigits * task + 64 + lane];
			taskOffsets[3] = taskScan_global[NumDigits * task + 96 + lane];
		}
	}

	// Loop over all blocks for this task. Read the packed scanned digit counts
	// and emit unpacked global scatter indices.
	for(int block(taskRange.x); block < taskRange.y; ++block) {
		uint packedIndex = block * NumChannels + lane;

		if(lane < NumChannels) {
			uint packed = blockLocalScan_global[packedIndex];
			column_shared[lane] = 0xffff & packed;
			column_shared[NumChannels + lane] = packed>> 16;
		}
		if(7 == NumBits) {
			uint packed = blockLocalScan_global[packedIndex + 32];
			column_shared[32 + lane] = 0xffff & packed;
			column_shared[96 + lane] = packed>> 16;			
		}

		// Get the local offsets for each digit.
		#pragma unroll
		for(int i = 0; i < CountsPerThread; ++i) {
			uint index = WARP_SIZE * i + lane;
			uint offset = column_shared[index];
			uint next = column_shared[index + 1];
			uint count = (next - offset);

			// Subtract out the scaled scan to accelerate global scatter in the
			// sort kernel.
			blockGlobalScan_global[NumDigits * block + index] = 
				taskOffsets[i] - offset;
			
			taskOffsets[i] += count;
		}
	}
}


#define GEN_SORTDOWNSWEEP(Name, NumBits, NumThreads, BlocksPerSM)			\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* taskScan_global, int taskQuot,						\
	int taskRem, int numBlocks, int blockSize,								\
	const uint* blockLocalScan_global, uint* blockGlobalScan_global) {		\
																			\
	SortDownsweep<NumBits, NumThreads>(taskScan_global, taskQuot,			\
		taskRem, numBlocks, blockSize, blockLocalScan_global,				\
		blockGlobalScan_global);											\
}
		
#define NUM_DOWNSWEEP_THREADS (WARP_SIZE * NUM_DOWNSWEEP_WARPS)

GEN_SORTDOWNSWEEP(SortDownsweep_1, 1, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_2, 2, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_3, 3, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_4, 4, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_5, 5, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_6, 6, NUM_DOWNSWEEP_THREADS, 8)
GEN_SORTDOWNSWEEP(SortDownsweep_7, 7, NUM_DOWNSWEEP_THREADS, 8)

