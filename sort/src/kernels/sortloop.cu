#pragma once

#include "sortlocal.cu"
#include "sortgeneric.cu"
#include "sortstore.cu"

DEVICE void AdvanceStreamOffsets(uint tid, uint numDigits, 
	uint* localScan_shared, uint* scaledGlobalOffsets_shared, 
	uint& globalOffsetAdvance, uint numValues) {

	if(tid < numDigits) {
		// Get this digit's local scan offset.
		uint index = (numDigits / 2 - 1) & tid;
		uint scan = localScan_shared[index];
		if(tid >= numDigits / 2) scan>>= 16;
		scan &= 0xffff;

		// Get the next digit's local scan offset.
		index = (numDigits / 2 - 1) & (tid + 1);
		uint scan2 = localScan_shared[index];
		if((tid + 1) >= numDigits / 2) scan2>>= 16;
		scan2 &= 0xffff;
		if(numDigits - 1 == tid) scan2 = 4 * numValues;

		uint count = scan2 - scan;

		// Subtract the scan offset from globalOffsetAdvance to get the 
		// scatter pointer offset.
		scaledGlobalOffsets_shared[tid] = globalOffsetAdvance - scan;

		// Advance the globalOffsetAdvance index by the digit count.
		globalOffsetAdvance += count;
	}

}

template<int NumThreads, int NumBits, bool LoadFromTexture>
DEVICE2 void SortInnerLoop(uint tid, uint block, const uint* keys_global_in, 
	const uint* blockLocalScan_global, uint texBlock, uint bit,
	uint* debug_global_out, uint* scattergather_shared, uint* scratch_shared,
	uint* scaledGlobalOffsets_shared, uint* blockLocalScan_shared,
	uint& globalOffsetAdvance, bool useFusedKey, Values keys,
	Values fusedKeys) {

	const int NumValues = VALUES_PER_THREAD * NumThreads;
	const int NumDigits = 1<< NumBits;
	const int NumChannels = NumDigits / 2;

	if(tid < NumChannels)
		blockLocalScan_shared[tid] = 
			blockLocalScan_global[NumChannels * block + tid];

	if(NumDigits >= WARP_SIZE) __syncthreads();

	AdvanceStreamOffsets(tid,NumDigits, blockLocalScan_shared, 
		scaledGlobalOffsets_shared, globalOffsetAdvance, NumValues);
	
	LoadKeysGlobal<NumThreads, NumBits, LoadFromTexture>(tid, block, 
		keys_global_in, texBlock, bit, scattergather_shared, useFusedKey, keys,
		fusedKeys);

	if(useFusedKey) bit = 24;
	bool loadFromArray = !LoadFromTexture;
	
	if(1 == NumBits) 
		SortAndScatter(tid, fusedKeys, bit, 1, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global_out);
	else if(2 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global_out);
	else if(3 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global_out);
	else if(4 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global_out);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global_out);
	} else if(5 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global_out);
		SortAndScatter(tid, fusedKeys, bit + 2, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global_out);
	} else if(6 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global_out);
		SortAndScatter(tid, fusedKeys, bit + 3, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global_out);
	} else if(7 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global_out);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, true,
			scattergather_shared, scratch_shared, debug_global_out);
		SortAndScatter(tid, fusedKeys, bit + 4, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global_out);
	}


}

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
	__shared__ uint blockLocalScan_shared[NumChannels];

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

		Values keys, fusedKeys;
		SortInnerLoop<NumThreads, NumBits, LoadFromTexture>(tid, block,
			keys_global_in, blockLocalScan_global, block, bit, debug_global_out,
			scattergather_shared, scratch_shared, scaledGlobalOffsets_shared, 
			blockLocalScan_shared, globalOffsetAdvance, 0 != ValueCount, keys, 
			fusedKeys);
	
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
