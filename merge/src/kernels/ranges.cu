#include "common.cu"

const uint ValueNull = 0xffffffff;


////////////////////////////////////////////////////////////////////////////////
// GetRangeBallot

// Before calling FillRangeBallot, the begin array is initialized to ValueNull.
// Each radix digit that occurs in the active array sets its first array index
// at begin[digit]. On output, begin/end STL-style pairs are generated. These
// provide lower_bound indices for insertion (begin) or upper_bound indices for
// insertion (end). This also works for empty digits.

DEVICE uint2 GetRangeBallot32(volatile uint* offsets_shared, uint lane, 
	uint count) {

	// We know which offset lanes are present. Find the last occupied lane
	// at or before the lane variable. This requires masking out this lane's
	// bit and all preceding bits, and using ctz to find the least-sig digit
	// set. This becomes the .end value for the lane.
	uint beginX = offsets_shared[lane];
	int pred = ValueNull != beginX;
	uint bits = __ballot(pred);
	uint mask = 0xfffffffe<< lane;
	uint endIndex = ctz(mask & bits);

	uint endX = (endIndex < WARP_SIZE) ? offsets_shared[endIndex] : count;
	beginX = pred ? beginX : endX;
	return make_uint2(beginX, endX);
}

// Return two pairs of ranges. (.x, .y) hold (begin, end) for the first half of
// the array. (.z, .w) hold (begin, end) for the second half of the array.
DEVICE uint4 GetRangeBallot64(volatile uint* offsets_shared, uint lane, 
	uint count) {

	// Each lane manages two radix digits.
	uint beginX = offsets_shared[lane];
	uint beginY = offsets_shared[WARP_SIZE + lane];
	int predX = ValueNull != beginX;
	int predY = ValueNull != beginY;
	uint bitsX = __ballot(predX);
	uint bitsY = __ballot(predY);

	uint mask = 0xfffffffe<< lane;
		
	// The low value (x) requires examination of both the low int (masked)
	// and the high int (unmasked).
	bitsX &= mask;
	uint endIndexXlow = ctz(bitsX);
	uint endIndexXhigh = ctz(bitsY);
	uint endIndexX = bitsX ? endIndexXlow : (WARP_SIZE + endIndexXhigh);
	uint endIndexY = ctz(mask & bitsY);
		
	uint endX = (endIndexX < 2 * WARP_SIZE) ? 
		offsets_shared[endIndexX] : count;
	uint endY = (endIndexY < WARP_SIZE) ? 
		offsets_shared[WARP_SIZE + endIndexY] : count;

	beginX = predX ? beginX : endX;
	beginY = predY ? beginY : endY;

	return make_uint4(beginX, endX, beginY, endY);
} 


////////////////////////////////////////////////////////////////////////////////
// RangeBinarySearch
// This function performs a lower_bound or upper_bound within the shared mem
// range. It closely follows the STL <algorithm> versions of these functions.

// pass kind = 0 for lower_bound, kind = 1 for upper_bound.
template<typename T>
DEVICE2 uint RangeBinarySearch(const T* values, uint begin, uint end, T key,
	int kind, int& count) {
	
	count = 0;
	while(end > begin) {
		// Keep a count for debugging and diagnostics.
		++count;

		uint mid = (begin + end) / 2;

		T midValue = values[mid];

		// Compare the value at the splitter with the key.
		// For lower_bound use <, because a tie should be false, as we want to
		// insert before other equal values. For upper_bound use <=, because a 
		// tie should be true, as we want to insert after other equal values.
		int pred = kind ? (midValue <= key) : (midValue < key);

		// If pred is true, recurse right.
		// If pred is false, recurse left.
		if(pred)
			begin = mid + 1;
		else
			end = mid;
	}
	return begin;
}


////////////////////////////////////////////////////////////////////////////////
// GetRadixShift finds the number of bits to right shift so that a numRadixBits
// mask can be taken to isolate only the most-significant bits that differ
// between the four endpoints. These may be the four endpoints for either a 
// a search or a merge. Note the endpoints are for the first and last values
// encountered during the loop iteration for the search or merge, not the first
// and last values encountered for the entire thread block. This gives us a 
// tight shift to narrow down the binary search range.

DEVICE2 int GetRadixShift(uint a0, uint a1, uint a2, uint a3, 
	int numRadixBits) {

	// Find the most significant bit which differs between the four values.
	uint bits = (a0 & a1 & a2 & a3) ^ (a0 | a1 | a2 | a3);
	int msb = __clz(bits);
	int shift = 8 * sizeof(uint) - msb - numRadixBits;

	// Shift right by this count and mask the bottom numRadixBits. This as 
	// accomplished with bfe.
	shift = max(shift, 0);
	return shift;
}

// Find the radix shift for 64-bit types. This first masks the 
DEVICE2 int GetRadixShift(uint2 a0, uint2 a1, uint2 a2, uint2 a3,
	int numRadixBits) {

	int shift;
	uint high = (a0.y & a1.y & a2.y & a3.y) ^ (a0.y | a1.y | a2.y | a3.y);
	if(high) {
		int msb = __clz(high);
		shift = 64 - msb - numRadixBits;
	} else {
		uint low = (a0.x & a1.x & a2.x & a3.x) ^ (a0.x | a1.x | a2.x | a3.x);
		int msb = __clz(low);
		shift = 32 - msb - numRadixBits;
	}
	return max(shift, 0);
}


////////////////////////////////////////////////////////////////////////////////
// Interval testing

// TestSearchInterval
// Analyzes a radix digit search interval. If an interval of repeated values is
// detected, return the most significant bit as set. This is OR'd into the 
// combined search interval code.
template<typename T>
DEVICE2 uint TestSearchInterval(const T* data_shared, uint2 range) {
	uint repeated = 0;
	if(range.y > range.x) {
		T first = data_shared[range.x];
		T last = data_shared[range.y - 1];
		if(first == last)
			repeated = 0x80000000;
	}
	return repeated;
}

// CombineSearchInterval
// Packs the begin and end ranges into a single 32-bit integer. Also adds in the
// repeated value flag.
DEVICE uint CombineSearchInterval(uint2 range, uint repeated) {
	uint combined = bfi(range.x, range.y, 16, 16);
	if(repeated) combined |= repeated;
	return combined;	
}


////////////////////////////////////////////////////////////////////////////////

// Search from 
template<typename T>
DEVICE2 uint SharedBinarySearch(T key, uint digit, const uint* ranges_shared,
	const T* aData, int kind, bool adjust) {




}


/*
DEVICE void FillRangeScan(volatile uint* begin, volatile uint* end, uint lane,
	uint numRadixSlots, uint count) {

	if(WARP_SIZE == numRadixSlots) {
		// Drag the start indices from the right to the left. These are the 
		// end indices of the range pair.
		uint endX = (WARP_SIZE - 1 == lane) ? count : begin[lane + 1];
		end[lane] = endX;
		
	//	return;
		uint endX2 = ValueNull;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			uint offset = 1<< i;
			uint lane2 = lane + offset;
			if(lane2 < WARP_SIZE) endX2 = end[lane2];
			endX = min(endX, endX2);
			end[lane] = endX;
		}

		uint beginX = begin[lane];
		if(ValueNull == beginX) beginX = endX;
		begin[lane] = beginX;

	} else if(2 * WARP_SIZE == numRadixSlots) {
	}
}

*/

const uint BlockSize = 1024;

template<int NumThreads, int NumRadixBits>
DEVICE2 void TestRanges(const uint* values_global, uint2* rangePairs_global) {
	const int NumRadixSlots = 1<< NumRadixBits;
	
	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint block = blockIdx.x;

	__shared__ uint shared[BlockSize + 1];
	__shared__ uint begin[NumRadixSlots], end[NumRadixSlots];

	// Initialize the begin array by filling it with nulls.
	if(tid < NumRadixSlots)
		begin[tid] = ValueNull;

	// Load the values and store into shared mem.
	uint x = values_global[tid];
	shared[tid + 1] = x;

	__syncthreads();

	// Grab the preceding value and compare to x.
	uint preceding = shared[tid];
	uint diff = !tid || ((tid < 977) && (preceding != x));

	// If this is the first value with this radix digit, store its index (tid)
	// into begin[digit].
	if(diff) begin[x] = tid;
	
	// Generate the range pairs.
	if(tid < WARP_SIZE) {
		if(32 == NumRadixSlots) {
			uint2 range = GetRangeBallot32(begin, lane, 977);
			begin[tid] = range.x;
			end[tid] = range.y;
		} else if(64 == NumRadixSlots) {
			uint4 range = GetRangeBallot64(begin, lane, 977);
			begin[tid] = range.x;
			end[tid] = range.y;
			begin[WARP_SIZE + tid] = range.z;
			end[WARP_SIZE + tid] = range.w;
		}
	}
	__syncthreads();

	if(tid < NumRadixSlots) {
		uint2 pair = make_uint2(begin[tid], end[tid]);
		rangePairs_global[tid] = pair;
	}
}
/*

extern "C" __global__
void TestRanges32(const uint* values_global, uint2* rangePairs_global) {
	TestRanges<BlockSize, 5>(values_global, rangePairs_global);
}


extern "C" __global__
void TestRanges64(const uint* values_global, uint2* rangePairs_global) {
	TestRanges<BlockSize, 6>(values_global, rangePairs_global);
}
*/
