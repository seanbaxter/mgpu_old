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

template<typename T>
DEVICE2 uint SearchIntervalNoRadix(uint tid, const T* aData_shared,
	SearchParams<T> params, int kind) {

	uint index;
	if(tid < params.bCount)
		index = RangeBinarySearch(aData_shared, 0, params.aCount, params.b, 
			kind);
	return index;
}

template<typename T, typename T2>
DEVICE2 uint SearchInterval(uint tid, const T* aData_shared, 
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

		int shift = GetRadixShift(aRadix, bRadix, rightRadixA, rightRadixB,
			numRadixBits);
		radixShift_shared = shift;
	}
	__syncthreads();

	uint shift = radixShift_shared;
	uint aDigit = bfe(aRadix, shift, numRadixBits);

	// Extract the radix digit from the preceding aData. If tid < aCount and
	// the preceding aDigit differs from tid's aDigit, set radix_shared[aDigit]
	// to indicate we have a starting offset for a radix digit. 
	uint aPrevDigit = bfe(ConvertToRadix(params.aPrev), shift, numRadixBits);
	if((tid < params.aCount) && (!tid || (aPrevDigit != aDigit)))
		radix_shared[aDigit] = tid;
	__syncthreads();

	// GetRangeBallot[32|64] ballot scans the offsets into STL-style begin/end
	// iterator pairs to limit the search range.
	if(tid < WARP_SIZE) {
		if(32 == NumRadixSlots) {
			uint2 range = GetRangeBallot32(radix_shared, tid, params.aCount);
			uint combined = CombineSearchInterval(range, 
				adjust ? TestSearchInterval(aData_shared, range) : 0);

			radix_shared[tid] = combined;

		} else if(64 == NumRadixSlots) {
			uint4 range = GetRangeBallot64(radix_shared, tid, params.aCount);

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


	uint index;

	// If the bData_shared is valid, perform a binary search.
	if(tid < params.bCount) {
		// Extract the bData digit and lookup the range for the corresponding 
		// aData digit.
		uint bDigit = bfe(bRadix, shift, numRadixBits);
		uint combined = radix_shared[bDigit];
		uint begin = 0x0000ffff & combined;
		
		if(adjust) {
			uint end = bfe(combined, 16, 15);
			uint end2 = (0x80000000 & combined) ? (begin + 1) : end;
			index = RangeBinarySearch(aData_shared, begin, end2, params.b,
				kind);
			if(index == end2) index = end;
		} else {
			uint end = combined>> 16;
			index = RangeBinarySearch(aData_shared, begin, end, params.b,
				kind);
		}
	}

	return index;
}


////////////////////////////////////////////////////////////////////////////////
// SearchInsertRange 

struct SearchInsertResult {
	uint index;
	int aConsume;
	int bConsume;
};

template<typename T, typename T2>
DEVICE2 SearchInsertResult SearchInsertRange(uint tid, int aCount, int bCount,
	int aRemaining, int bRemaining, T aData, T bData, T* aData_shared, 
	T* bData_shared, uint* radix_shared, int numRadixBits, int kind) {


	////////////////////////////////////////////////////////////////////////////
	// Grab the SearchParams and test if we can stream out all aData or all
	// bData values. If not, call SearchInterval.

	SearchInsertResult result;
	result.aConsume = aCount;
	result.bConsume = bCount;

	SearchParams<T> params;

	__shared__ int consumed_shared;

	if(aCount && bCount) {

		// Load the last value of both arrays to establish the consumed
		// counts.
		params.a = aData;
		params.b = bData;
		params.aLast = aData_shared[aCount - 1];
		params.bLast = bData_shared[bCount - 1];
		params.aPrev = aData_shared[tid - 1];
		params.bPrev = bData_shared[tid - 1];

		// Because we're only dealing with fragments of the datasets, we
		// need to be careful on which values can be inserted. Take the
		// upper_bound case as an example:
		
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 2 3 4 7 8 8 9 ...
		// bData = 1 2 2 2 6 6 9 9 ...

		// We can insert elements with values 1 - 6 from bData without 
		// ambiguity. However we can't be sure where to insert the pair of
		// 9s. The upper_bound semantics demand that they be inserted AFTER
		// all occurences of the same value in the aData stream. Because we
		// haven't seen the next elements in aData we don't know if we can
		// insert bData[6] and bData[7] at i = 8.

		// The same problem occurs with lower_bound: 
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 2 3 4 7 7 7 8 ... 
		// bData = 1 2 2 2 6 9 9 9 ...

		// aData[8] could be an 8, in which case it would precede bData[5],
		// or it could be a 9 or something larger, in which case the 9 terms
		// from bData should come first.

		// The problem is symmetric even when we only want to find insert 
		// points from bData into aData:
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 3 3 3 5 6 7 7 ...
		// bData = 1 1 2 3 3 3 3 4 (4 5 5 6 7 7 7)

		// The goal is to consume both the aData and bData arrays in shared
		// memory every iteration. However we need to retain aData values
		// when they are required to know where to insert upcoming bData
		// values. In the example above, we can consume all 8 values in 
		// bData, but only the first 4 values in aData. The remaining aData
		// values are shifted forward in aData_shared and are used for
		// supporting the search with the subsequent bData values.

		// When searching with lower_bound, if both aData and bData shared
		// memory arrays end with the same value, both arrays are completely
		// consumed.

		// At least one of the arrays will be completely consumed. To
		// summarize, when inserting bData into aData, lower_bound favors
		// consuming bData and upper_bound favors consuming aData.

		if(aRemaining + bRemaining) {
			// Compare the last element in bData to the last element in aData.
			bool pred = kind ? (params.bLast < params.aLast) : 
				(params.bLast <= params.aLast);

			// Get the preceding keys to find the first value that violates the
			// consume conditions.
			if(pred) {
				// If the last element in bData is < (or <=) the last element in
				// aData, completely consume the bData_shared stream.

				// If upper_bound consume all values in aData that are <= bLast.
				// If lower_bound consume all values in aData that are < bLast.
				bool inRange = kind ? (params.a <= params.bLast) :
					(params.a < params.bLast);
				bool inRangePrev = kind ? (params.aPrev <= params.bLast) :
					(params.aPrev < params.bLast);
				if(!tid) inRangePrev = true;
				if(!inRange && inRangePrev)
					consumed_shared = tid;

				__syncthreads();
				result.aConsume = consumed_shared;
			} else {
				// If the last element in bData is not < (or <=) the last
				// element in aData, completely consume the aData_shared stream.

				// If upper_bound consume all values in bData that are <= aLast.
				// If lower_bound consume all values in bData that are < aLast.
				bool inRange = kind ? (params.b < params.aLast) :
					(params.b <= params.aLast);
				bool inRangePrev = kind ? (params.bPrev < params.aLast) :
					(params.bPrev <= params.aLast); 
				if(!tid) inRangePrev = true;
				if(!inRange && inRangePrev)
					consumed_shared = tid;

				__syncthreads();
				result.bConsume = consumed_shared;
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////
	// Stream the values based on the consumed counts (params.aCount and 
	// params.bCount).

	if(!result.aConsume) {
		// We've run out of aData keys to merge into. All the keys get inserted
		// to the start of the current section.
		result.index = 0;
	} else if(result.bConsume) {
		// Both aCount and bCount are defined. For this we perform the batched
		// binary search.
		params.aCount = result.aConsume;
		params.bCount = result.bConsume;
		if(numRadixBits)
			result.index = SearchInterval<T, T2>(tid, aData_shared,
				bData_shared, params, radix_shared, numRadixBits, kind, true);
		else
			result.index = SearchIntervalNoRadix(tid, aData_shared, params,
				kind);
	}
	return result;
}


////////////////////////////////////////////////////////////////////////////////
// SearchBlock
// Run a lower_bound or upper_bound from a set of sorted keys.

template<int NumThreads, int NumRadixBits, typename T, typename T2>
DEVICE2 void SearchBlock(const T* aData_global, int2 aRange,
	const T* bData_global, int2 bRange, int kind, int2* indices_global) {

	const int NumRadixDigits = 1<< NumRadixBits;
	__shared__ uint radix_shared[NumRadixDigits];
	__shared__ T aData_shared[NumThreads], bData_shared[NumThreads];
	
	uint tid = threadIdx.x;

	// Poke the radix_shared array to make sure the compiler keeps it in.
	if(!tid) radix_shared[0] = 0;

	// aRange.x and bRange.y refer to the first element of each array in each
	// inner loop. They are not pointers to the next value to load. This
	// function streams at 

	// aLoaded and bLoaded are the counts of loaded but unprocessed values from
	// each stream.
	// aRemaining and bRemaining are the counts of unprocessed values from each
	// stream, including the loaded values.
	int aLoaded = 0;
	int bLoaded = 0;
	int aRemaining = aRange.y - aRange.x;
	int bRemaining = bRange.y - bRange.x;

	SearchInsertResult result;

	while(aRemaining || bRemaining) {
		
		////////////////////////////////////////////////////////////////////////
		// Load the values to fill the buffers.

		int aCount = min(NumThreads, aRemaining);
		int bCount = min(NumThreads, bRemaining);

		T a = (tid >= aLoaded) ? 
			aData_global[min(aRange.y - 1, aRange.x + tid)] : 
			aData_shared[tid + result.aConsume];
		T b = (tid >= bLoaded) ?
			bData_global[min(bRange.y - 1, bRange.x + tid)] :
			bData_shared[tid + result.bConsume];
		__syncthreads();

		aData_shared[tid] = a;
		bData_shared[tid] = b;
		__syncthreads();

		result = SearchInsertRange<T, T2>(tid, aCount, bCount, 
			aRemaining - aCount, bRemaining - bCount, a, b,
			aData_shared, bData_shared, radix_shared, NumRadixBits, kind);

		if(tid < result.bConsume)
			indices_global[bRange.x + tid] = 
				make_int2(result.index + aRange.x, b);


		////////////////////////////////////////////////////////////////////////
		// Advance the iterators by the consumed counts.

		// Update the offsets for the next serialization.
		aRange.x += result.aConsume;
		bRange.x += result.bConsume;
		
		aLoaded = aCount - result.aConsume;
		bLoaded = bCount - result.bConsume;

		aRemaining -= result.aConsume;
		bRemaining -= result.bConsume;
	}
}

extern "C" __global__ __launch_bounds__(256, 4)
void SearchBlockInt(const uint* aData_global, int2 aRange,
	const uint* bData_global, int2 bRange, int2* indices_global) {

	SearchBlock<256, 0, uint, uint>(aData_global, aRange, bData_global,
		bRange, 0, indices_global);
}
