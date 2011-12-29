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
	uint aDigit = bfe(aRadix, shift, NumRadixBits);

	// Extract the radix digit from the preceding aData. If tid < aCount and
	// the preceding aDigit differs from tid's aDigit, set radix_shared[aDigit]
	// to indicate we have a starting offset for a radix digit. 
	uint aPrevDigit = bfe(ConvertToRadix(params.aPrev), shift, NumRadixBits);
	if((tid < params.aCount) && (!tid || (aPrevDigit != aDigit)))
		radix_shared[digitA] = tid;
	__syncthreads();

	// GetRangeBallot[32|64] ballot scans the offsets into STL-style begin/end
	// iterator pairs to limit the search range.
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


template<int NumThreads, int NumRadixBits, typename T, typename T2>
DEVICE2 void SearchBlock(const T* aData_global, int2 aRange,
	const T* bData_global, int2 bRange, int kind) {

	const int NumRadixDigits = 1<< NumRadixBits;
	__shared__ uint radix_shared[NumRadixDigits];
	__shared__ T aData_shared[NumThreads], bData_shared[NumThreads];
	__shared__ uint indices_shared[2 * NumThreads];

	// aLoaded and bLoaded are the counts of loaded but unprocessed values from
	// each stream.
	// aRemaining and bRemaining are the counts of unprocessed values from each
	// stream, including the loaded values.
	int aLoaded = 0;
	int bLoaded = 0;
	int aRemaining = aRange.y - aRange.x;
	int bRemaining = bRange.y - bRange.x;

	while(aRemaining || bRemaining) {
		
		////////////////////////////////////////////////////////////////////////
		// Load the values to fill the buffers.

		
		
		int aLoad = min(aRemaining, NumThreads - aLoaded);
		int bLoad = min(bRemaining, NumThreads - bLoaded);



	}
	




	T preceding = aData_shared[tid - 1];
	result.a = aData_shared[tid];
	result.b = bData_shared[tid];
	result.aLast = aData_shared[aCount - 1];
	result.bLast = bData_shared[bCount - 1];
	*/

template<typename T>
struct SearchParams {
	T a, b;
	uint aCount, bCount;
	T aPrev, bPrev;
	T aLast, bLast;
};

template<int NumRadixBits, typename T, typename T2>
DEVICE2 uint SearchInterval(uint tid, uint lane, const T* aData_shared, 
	const T* bData_shared, SearchParams<T> params, uint* radix_shared, 
	int numRadixBits, int kind, bool adjust) {


	__shared__ T 
