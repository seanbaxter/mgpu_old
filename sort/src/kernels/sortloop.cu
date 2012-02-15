#pragma once

#include "sortlocal.cu"
#include "sortgeneric.cu"
#include "sortstore.cu"


////////////////////////////////////////////////////////////////////////////////
// SortFuncLoop

template<int NumThreads, int NumBits, int ValueCount, bool LoadFromTexture>
DEVICE2 void SortFuncLoop(const uint* keys_global_in, 
	const uint* blockLocalScan_global, const uint* taskOffsets_global,
	uint bit, uint* keys_global_out, uint taskQuot, uint taskRem, 
	uint numValueStreams, uint* debug_global_out, 
	const uint* values1_global_in, const uint* values2_global_in,
	const uint* values3_global_in, const uint* values4_global_in,
	const uint* values5_global_in, const uint* values6_global_in,
	// For VALUE_TYPE_MULTI, we have to pass each of the pointers in as 
	// individual arguments, not as arrays, or CUDA generates much worse code,
	// full of unified addressing instructions.
	uint* values1_global_out, uint* values2_global_out,
	uint* values3_global_out, uint* values4_global_out,
	uint* values5_global_out, uint* values6_global_out) {
	
	const int NumValues = VALUES_PER_THREAD * NumThreads;

	const int NumDigits = 1<< NumBits;
	const int NumChannels = NumDigits / 2;

	const int NumWarps = NumThreads / WARP_SIZE;

	const int Stride = WARP_SIZE + 1;
	// const int ScratchSize = 2 * (NumThreads + NumWarps) + 4 * WARP_SIZE + 32;
	const int ScratchSize = 2 * (NumThreads + NumWarps) +
		2 * (WARP_SIZE + WARP_SIZE / 2);

	__shared__ uint scattergather_shared[NumWarps * VALUES_PER_THREAD * Stride];
	__shared__ uint scratch_shared[ScratchSize];

	__shared__ uint scaledGlobalOffsets_shared[NumDigits];
	__shared__ uint localScan_shared[NumChannels];

	uint tid = threadIdx.x;
	uint task = blockIdx.x;

	int2 range = ComputeTaskRange(task, taskQuot, taskRem);

	uint globalOffsetAdvance;
	if(tid < NumDigits)
		// Load the global offset for this digit.
		globalOffsetAdvance = 4 * taskOffsets_global[NumDigits * task + tid];


	////////////////////////////////////////////////////////////////////////////
	// Loop over every block in the range.
	
	for(int block(range.x); block < range.y; ++block) {
	
		////////////////////////////////////////////////////////////////////////
		// Load the packed local scan for this block and update global scatter
		// pointers.

		if(tid < NumChannels)
			localScan_shared[tid] = 
				blockLocalScan_global[NumChannels * block + tid];
		if(NumDigits > WARP_SIZE) __syncthreads();

		if(tid < NumDigits) {
			// Get this digit's local scan offset.
			uint index = (NumDigits / 2 - 1) & tid;
			uint scan = localScan_shared[index];
			if(tid >= NumDigits / 2) scan>>= 16;
			scan &= 0xffff;

			// Get the next digit's local scan offset.
			index = (NumDigits / 2 - 1) & (tid + 1);
			uint scan2 = localScan_shared[index];
			if((tid + 1) >= NumDigits / 2) scan2>>= 16;
			scan2 &= 0xffff;
			if(NumDigits - 1 == tid) scan2 = 4 * NumValues;

			uint count = scan2 - scan;

			// Subtract the scan offset from globalOffsetAdvance to get the 
			// scatter pointer offset.
			scaledGlobalOffsets_shared[tid] = globalOffsetAdvance - scan;

			// Advance the globalOffsetAdvance index by the digit count.
			globalOffsetAdvance += count;
		}

		////////////////////////////////////////////////////////////////////////
		// Load keys and run the local sort.

		Values keys, fusedKeys;
		LoadAndSortLocal<NumThreads, NumBits, LoadFromTexture>(tid, block, 
			keys_global_in, block, bit, debug_global_out, scattergather_shared,
			scratch_shared, 0 != ValueCount, keys, fusedKeys);


		////////////////////////////////////////////////////////////////////////
		// Store the keys and values to global memory.

		if(0 == ValueCount) {

			// Store only keys.
			GatherBlockOrder(tid, false, NumThreads, keys, 
				scattergather_shared);

			ScatterKeysSimple(tid, keys_global_out, bit, NumBits, 
				scaledGlobalOffsets_shared, keys);
		}
	}
}


////////////////////////////////////////////////////////////////////////////////

#define GEN_SORT_LOOP(Name, NumThreads, NumBits, ValueCount,				\
	LoadFromTexture, BlocksPerSM)											\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global_in, const uint* blockLocalScan_global,	\
	const uint* taskOffsets_global, uint bit, uint* keys_global_out,		\
	uint taskQuot, uint taskRem, uint* debug_global_out) {					\
																			\
	SortFuncLoop<NumThreads, NumBits, ValueCount, LoadFromTexture>(			\
		keys_global_in, blockLocalScan_global, taskOffsets_global,			\
		bit, keys_global_out, taskQuot, taskRem, 0, debug_global_out,		\
		0, 0, 0, 0, 0, 0,													\
		0, 0, 0, 0, 0, 0);													\
}
