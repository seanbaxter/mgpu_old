#pragma once

#include "sortscan1.cu"
#include "sortmultiscan.cu"

template<int ValuesPerThread>
struct StridedDenom {
	static const int Denom = WARP_SIZE;
};
template<> struct StridedDenom<24> {
	static const int Denom = 8;
};

// Currently only supports a pow2 number of values per thread or 24.
template<int NumThreads, int ValuesPerThread, int NumBits>
struct LocalSortSize {
	static const int NumValues = NumThreads * ValuesPerThread;
	static const int Multipass = NumBits > 5;
	static const int LargestBitPass = (NumBits > 5) ? (NumBits - 3) : NumBits;

	static const int ScratchSize = 
		MultiscanParams<NumThreads, LargestBitPass>::ScratchSize;
	
	static const bool IsPower2 = IS_POW_2(ValuesPerThread);
	static const int StridedSize = NumValues + 
		NumValues / StridedDenom<ValuesPerThread>::Denom;

	static const int ScatterGatherSize = Multipass ? StridedSize : NumValues;

	static const int SharedSize = MAX(ScratchSize, ScatterGatherSize);
};


////////////////////////////////////////////////////////////////////////////////
// ScatterGatherStrided
// Takes ValuesPerThread keys and scaled scatter indices (pre-multiplied by 4),
// scatters the keys to shared memory, and reads them back in thread order
// without bank conflict. This is for multi-pass local sorts.

template<int ValuesPerThread>
DEVICE2 void ScatterGatherStrided(uint tid, uint NumThreads, uint* keys,
	const uint* scatter, volatile uint* shared) {

	if(IS_POW_2(ValuesPerThread)) {

		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v) {
			uint index = scatter[v];
			index = shr_add(index, LOG_WARP_SIZE, index);
			index &= ~3;

			StoreShifted(shared, index, keys[v]);
		}
		__syncthreads();

		int offset = ValuesPerThread * tid;
		offset = shr_add(offset, LOG_WARP_SIZE, offset);
		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v)
			keys[v] = shared[offset + v];
	
	} else if(24 == ValuesPerThread) {
		
		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v) {
			uint index = scatter[v];
			
			// divide by 8.
			index = shr_add(index, 3, index);
			index &= ~3;

			StoreShifted(shared, index, keys[v]);
		}
		__syncthreads();

		// Since the warp size is not a multiple of the values per thread, the
		// spaces can come inside the runs. We have to do three runs of 8 values
		// and calculate the offset at the start of each run.

		int index = ValuesPerThread * tid;
		index = shr_add(index, 3, index);

		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v)	
			keys[v] = shared[index + v + v / 8];
		
	}
	__syncthreads();
}

template<int ValuesPerThread>
DEVICE2 void ScatterGatherUnstrided(uint tid, uint NumThreads, uint* keys,
	const uint* scatter, volatile uint* shared) {

	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		StoreShifted(shared, scatter[v], keys[v]);
	__syncthreads();

	for(int v(0); v < ValuesPerThread; ++v)
		keys[v] = shared[NumThreads * v + tid];
}




////////////////////////////////////////////////////////////////////////////////
// SortLocal

template<int NumThreads, int NumBits, int ValuesPerThread>
DEVICE2 void SortLocal(uint tid, uint* keys, uint bit, 
	volatile uint* shared, bool recalcDigits) {

	uint scatter[ValuesPerThread];
	if(1 == NumBits) {
		MultiScanCounters<NumThreads, 1, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
	//	SortScatter1<NumThreads, ValuesPerThread>(tid, keys, bit, shared,
	//		scatter, recalcDigits);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(2 == NumBits) {
		MultiScanCounters<NumThreads, 2, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(3 == NumBits) {
		MultiScanCounters<NumThreads, 3, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(4 == NumBits) {
		MultiScanCounters<NumThreads, 4, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(5 == NumBits) {
		MultiScanCounters<NumThreads, 5, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(6 == NumBits) {
		MultiScanCounters<NumThreads, 3, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherStrided<ValuesPerThread>(tid, NumThreads, keys, scatter,
			shared);
		MultiScanCounters<NumThreads, 3, ValuesPerThread>(tid, keys, 3 + bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	} else if(7 == NumBits) {
		MultiScanCounters<NumThreads, 4, ValuesPerThread>(tid, keys, bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherStrided<ValuesPerThread>(tid, NumThreads, keys, scatter,
			shared);
		MultiScanCounters<NumThreads, 3, ValuesPerThread>(tid, keys, 4 + bit,
			shared, scatter, false, recalcDigits, false);
		ScatterGatherUnstrided<ValuesPerThread>(tid, NumThreads, keys,
			scatter, shared);
	}
}

