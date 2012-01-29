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

DEVICE2 uint ComputeBucketCounts(const Values digits, uint numBits,
	uint2& bucketsPacked, uint2& offsetsPacked) {

	uint predInc = 0;
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

DEVICE2 uint2 ComputeBucketCounts16(const Values digits, uint& bucketsPacked,
	uint2& offsetsPacked) {

	uint64 predInc = 0;
	const int BitsPerValue = 4;
		
	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		uint digit = digits[v];
		uint shift = BitsPerValue * digit;

		// Insert the previous predInc to bucketsPacked.
		// Don't need to clear the high bits because bfi will do it
		uint prevPredInc = (uint)(predInc>> shift);

		if(0 == v) {
			// set predInc with shift
			predInc = 1<< shift;
			offsetsPacked.x = 0;
			bucketsPacked = digit;
		} else if(v < 4) {
			// bfi generates better code than shift and OR
			offsetsPacked.x = bfi(offsetsPacked.x, prevPredInc, 8 * v,
				BitsPerValue);
			bucketsPacked = bfi(bucketsPacked, digit, 4 * v, 4);
		} else if(4 == v) {
			// If we're processing 3 bits we have to clear out the high bits of
			// prevPredInc, because otherwise they won't be overwritten to zero
			// by bfi.
			offsetsPacked.y = prevPredInc;
			bucketsPacked = bfi(bucketsPacked, digit, 4 * v, 4);
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, prevPredInc, 8 * (v - 4),
				BitsPerValue);
			bucketsPacked = bfi(bucketsPacked, digit, 4 * v, 4);
		}

		if(v) predInc += 1ull< shift;
	}
	return __ulonglong2hilouint2(predInc);
}


////////////////////////////////////////////////////////////////////////////////
// FindScatterIndices is the function for performing a sort over a single
// digit of 1, 2, or 3 bits. It returns eight scatter indices, packed into four
// ints. 

#define SCAN3_1WARP
#define SCAN4_1WARP

DEVICE void FindScatterIndices(uint tid, Values digits, uint numBits, 
	uint numThreads, uint* scratch_shared, uint packed[4], uint* debug_global) {

	if(1 == numBits) {
		SortScatter1(tid, digits, numThreads, packed, scratch_shared, 
			debug_global);
	} else if(2 == numBits) {
		uint2 bucketsPacked, offsetsPacked;
		uint predInc = ComputeBucketCounts(digits, 2, bucketsPacked, 
			offsetsPacked);

		uint2 scanOffsets = MultiScan2(tid, predInc, numThreads, scratch_shared,
			debug_global);

		SortScatter2_8(scanOffsets, bucketsPacked, offsetsPacked, packed);

	} else if(3 == numBits) {
		uint2 bucketsPacked, offsetsPacked;
		uint predInc = ComputeBucketCounts(digits, 3, bucketsPacked, 
			offsetsPacked);

		uint4 scanOffsets;
#ifdef SCAN3_1WARP
		scanOffsets = MultiScan3_1Warp(tid, Expand8Uint4To8Uint8(predInc),
			numThreads, bucketsPacked, offsetsPacked, scratch_shared, 
			debug_global);
#endif
#ifdef SCAN3_2WARP
		scanOffsets = MultiScan3_2Warp(tid, Expand8Uint4To8Uint8(predInc),
			numThreads, bucketsPacked, offsetsPacked, scratch_shared, 
			debug_global);
#endif

		SortScatter3_8(scanOffsets, bucketsPacked, offsetsPacked, packed);

	} else if(4 == numBits) {
		uint bucketsPacked;
		uint2 offsetsPacked;
		uint2 predInc = ComputeBucketCounts16(digits, bucketsPacked, 
			offsetsPacked);

		uint2 predIncLow = Expand8Uint4To8Uint8(predInc.x);
		uint2 predIncHigh = Expand8Uint4To8Uint8(predInc.y);
		uint4 predInc4 = make_uint4(predIncLow.x, predIncLow.y,
			predIncHigh.x, predIncHigh.y);

		uint4 scanOffsetsLow, scanOffsetsHigh;
		uint2 localOffsets2;

#ifdef SCAN4_1WARP
		MultiScan4_1Warp(tid, predInc4, numThreads, bucketsPacked, 
			localOffsets2, scanOffsetsLow, scanOffsetsHigh, scratch_shared,
			debug_global);
#endif
	}
}
