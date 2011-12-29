#include "ranges.cu"

// NOTE: merge from exact ranges.

// build prologue code to binary search to establish ranges in the same kernel.


template<typename T>
struct SearchParams {
	T a, b;
	uint aCount, bCount;
	T aPrev, bPrev;
	T aLast, bLast;
};

template<typename T, typename T2>
DEVICE2 uint SearchInterval(uint tid, uint lane, const T* aData_shared, 
	const T* bData_shared, SearchParams<T> params, uint* radix_shared, 
	int numRadixBits, int kind, bool adjust) {

	const int NumRadixSlots = 1<< numRadixBits;

	// Start by clearing the radix digit offsets.
	if(tid < NumRadixSlots)
		radix_shared[tid] = ValueNull;

	T2 aRadix = ConvertToRadix(params.a);
	T2 bRadix = ConvertToRadix(params.b);

	__shared__ uint radixShift_shared;

	// Have the first thread set the radix shift. It's better to do this on a 
	// single thread, sync, and propagate the count through shared memory than
	// to do it in each thread. This is because The first thread already has to
	// load the first values from both arrays. We have an overall reduction for
	// shared mem access this way, plus it only runs from a single warp.
	if(!tid) {
		T2 rightRadixA = ConvertToRadix(params.aLast);
		T2 rightRadixB = ConvertToRadix(params.bLast);

		int shift = GetRadiXShift(aRadix, bRadix, rightRadixA, rightRadixB,
			numRadixBits);
		radixShift_shared = shift;
	}
	__syncthreads();

	uint shift = radixShift_shared;
	
	// TODO: does this work or do we need to shift? 
	uint digitA = bfe(aRadix, shift, NumRadixBits);

	// Get the value to the left.
	uint aPrevDigit = bfe(ConvertToRadix(params.aPrev), shift, NumRadixBits);

	// Set tid to the radix offset array if it holds the first value with the
	// radix digit.
	if(!tid || ((tid < aCount) && (aPrevDigit != digitA)))
		radix_shared[digitA] = tid;
	__syncthreads();

	// Calculate the search intervals for aData_shared and store in
	// radix_shared.data.
	if(tid < WARP_SIZE) {
		if(32 == NumRadixSlots) {
			uint2 range = GetRangeBallot32(radix_shared, lane, aCount);
			uint combined = CombineSearchInterval(range, 
				adjust ? TestSearchInterval(aData_shared, range) : 0);

			radix_shared[tid] = combined;
			
		} else if(64 == NumRadixSlots) {
			uint4 range = GetRangeBallot64(radix_shared, lane, aCount);

			uint2 range1 = make_uint2(range.x, range.y);
			uint2 range2 = make_uint2(range.z, range.w);

			uint combined1 = CombineSearchInterval(range1,
				adjust ? TestSearchInterval(aData_shared, range1) : 0);
			uint combined2 = CombineSearchInterval(range2,
				adjust ? TestSearchInterval(aData_shared, range2) : 0);

			radix_shared[tid] = combined1;
			radix_shared[WARP_SIZE + tid] = combined2;
		}
	}
	__syncthreads();

	uint index = ValueNull;

	// If the bData_shared is valid, perform a binary search.
	if(tid < bCount) {
		// Extract the bData digit and lookup the range for the corresponding 
		// aData digit.
		uint bDigit = bfe(bRadix, shift, NumRadixBits);
		uint combined = radix_shared[bDigit];
		uint begin = 0x0000ffff & combined;
		
		if(adjust) {
			uint end = bfe(combined, 16, 15);
			uint end2 = (0x80000000 & combined) ? (begin + 1) : end;
			index = RangeBinarySearch(aData_shared, begin, end2, kind);
			if(index == end2) index = end;
		} else {
			uint end = combined>> 16;
			index = RangeBinarySearch(aData_shared, begin, end, kind);
		}
	}

	return result;
}

/*
template<int NumThreads, int NumRadixBits, typename T, typename T2>

	T preceding = aData_shared[tid - 1];
	result.a = aData_shared[tid];
	result.b = bData_shared[tid];
	result.aLast = aData_shared[aCount - 1];
	result.bLast = bData_shared[bCount - 1];
	*/