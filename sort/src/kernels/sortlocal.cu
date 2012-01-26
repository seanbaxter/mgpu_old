#pragma once

#include "sortscan1.cu"
#include "sortscan2.cu"
#include "sortscan3.cu"

// Load 8 fused keys and produce a bucket total and a key offset within
// the thread-bucket. If 4 buckets are being counted, predInc is packed
// with bytes. If 8 buckets are being counted, predInc is packed with
// nibbles. In both cases, offsetsPacked is filled with bytes. This is
// because we'd have to expand the offsets to bytes anyway, so we do it
// here rather than first packing as nibbles.

// returns predInc packed into nibbles (for numBits=3) or bytes (numBits = 2).

template<typename T>
DEVICE2 T ComputeBucketCounts(const Values digits, uint numBits,
	uint2& bucketsPacked, uint2& offsetsPacked) {

	T predInc = 0;
	const int BitsPerValue = (2 == numBits) ? 8 : 4;
		
	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		uint digit = digits[v];
		uint shift = BitsPerValue * digit;

		// Insert the previous predInc to bucketsPacked.
		// Don't need to clear the high bits because bfi will do it
		uint prevPredInc = predInc>> shift;

		if(0 == v) {
			// set predInc with shift
			predInc = 1<< shift;
			offsetsPacked.x = 0;
			bucketsPacked.x = digit;
		} else if(v < 4) {
			// bfi generates better code than shift and OR
			offsetsPacked.x = bfi(offsetsPacked.x, prevPredInc, 8 * v,
				BitsPerValue);
			bucketsPacked.x = bfi(bucketsPacked.x, digit, 4 * v, 4);
		} else if(4 == v) {
			// If we're processing 3 bits we have to clear out the high bits of
			// prevPredInc, because otherwise they won't be overwritten to zero
			// by bfi.
			if(3 == numBits) prevPredInc &= 0x0f;
			offsetsPacked.y = prevPredInc;
			bucketsPacked.y = digit;
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, prevPredInc, 8 * (v - 4),
				BitsPerValue);
			bucketsPacked.y = bfi(bucketsPacked.y, digit, 4 * (v - 4), 4);
		}

		if(v) predInc = shl_add(1, shift, predInc);
	}
	return predInc;
}


////////////////////////////////////////////////////////////////////////////////
// FindScatterIndices is the function for performing a sort over a single
// digit of 1, 2, or 3 bits. It returns eight scatter indices, packed into four
// ints. 

DEVICE void FindScatterIndices(uint tid, Values digits, uint numBits, 
	uint numThreads, uint* scratch_shared, uint packed[4], uint* debug_global) {

	if(1 == numBits) {
		SortScatter1(tid, digits, numThreads, packed, scratch_shared, 
			debug_global);
	} else if(2 == numBits) {
		uint2 bucketsPacked, offsetsPacked;
		uint predInc = ComputeBucketCounts<uint>(digits, 2, bucketsPacked, 
			offsetsPacked);

		uint2 scanOffsets = MultiScan2(tid, predInc, numThreads, scratch_shared,
			debug_global);

		SortScatter2_8(scanOffsets, bucketsPacked, offsetsPacked, packed);

	} else if(3 == numBits) {
		uint2 bucketsPacked, offsetsPacked;
		uint predInc = ComputeBucketCounts<uint>(digits, 3, bucketsPacked, 
			offsetsPacked);

		uint4 scanOffsets = MultiScan3(tid, Expand8Uint4To8Uint8(predInc),
			numThreads, bucketsPacked, offsetsPacked, scratch_shared, 
			debug_global);

		SortScatter3_8(scanOffsets, bucketsPacked, offsetsPacked, packed);
	} else if(4 == numBits) {
		uint2 bucketsPacked, offsetsPacked;
		uint64 predInc = ComputeBucketCounts<uint64>(digits, 4, bucketsPacked, 
			offsetsPacked);



	}
}

