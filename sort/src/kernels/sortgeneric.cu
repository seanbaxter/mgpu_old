// Load 8 fused keys and produce a bucket total and a key offset within
// the thread-bucket. If 4 buckets are being counted, predInc is packed
// with bytes. If 8 buckets are being counted, predInc is packed with
// nibbles. In both cases, offsetsPacked is filled with bytes. This is
// because we'd have to expand the offsets to bytes anyway, so we do it
// here rather than first packing as nibbles.

// returns predInc packed into nibbles (for NUM_BITS=3) or bytes (NUM_BITS = 2).

DEVICE uint ComputeFusedKeyTotals(const Values fusedKeys, uint bit, 
	uint numBits, uint2& bucketsPacked, uint2& offsetsPacked) { 

	uint predInc = 0;
	const int BitsPerValue = (2 == numBits) ? 8 : 4;
		
	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		// Note: if this is the first pass for a fused key sort, there is likely
		// optimization potential for exploiting knowledge that bit = 0.
		uint bucket = bfe(fusedKeys[v], bit, numBits);
		uint shift = BitsPerValue * bucket;

		// Insert the previous predInc to bucketsPacked.
		// Don't need to clear the high bits because bfi will do it
		uint prevPredInc = predInc>> shift;

		if(0 == v) {
			// set predInc with shift
			predInc = 1<< shift;
			offsetsPacked.x = 0;
			bucketsPacked.x = bucket;
		} else if(v < 4) {
			// bfi generates better code than shift and OR
			offsetsPacked.x = bfi(offsetsPacked.x, prevPredInc, 8 * v,
				BitsPerValue);
			bucketsPacked.x = bfi(bucketsPacked.x, bucket, 4 * v, 4);
		} else if(4 == v) {
			// If we're processing 3 bits we have to clear out the high bits of
			// prevPredInc, because otherwise they won't be overwritten to zero
			// by bfi.
			if(3 == numBits) prevPredInc &= 0x0f;
			offsetsPacked.y = prevPredInc;
			bucketsPacked.y = bucket;			
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, prevPredInc, 8 * (v - 4),
				BitsPerValue);
			bucketsPacked.y = bfi(bucketsPacked.y, bucket, 4 * (v - 4), 4);
		}

		if(v) predInc = shl_add(1, shift, predInc);
	}
	return predInc;
}


#include "sortscan1.cu"
#include "sortscan2.cu"
#include "sortscan3.cu"

// Read fused keys from shared memory, scan, and scatter the fused keys into
// strided shared memory. 

DEVICE void SortAndScatter(uint tid, Values fusedKeys, uint bit, uint numBits,
	bool loadKeysFromArray, uint* debug_global) {

	uint packed[4];

	if(loadKeysFromArray) {	
		volatile uint* threadData = scattergather_shared + 
			StridedThreadOrder(tid * VALUES_PER_THREAD);
			
		#pragma unroll
		for(int v = 0; v < 8; ++v)
			fusedKeys[v] = threadData[v];
	}
	
	if(1 == numBits) {
		SortScatter1(tid, fusedKeys, bit, packed, 0, 0, 0, debug_global);

	} else if(2 == numBits) {
		uint2 bucketsPacked;
		uint2 offsetsPacked;
		uint predInc = ComputeFusedKeyTotals(fusedKeys, bit, 2, bucketsPacked, 
			offsetsPacked);

		uint2 scanOffsets;
		scanOffsets = MultiScan2(tid, predInc, 0, 0, 0, debug_global);

		SortScatter2_8(scanOffsets, bucketsPacked, offsetsPacked, fusedKeys,
			packed, tid);

	} else if(3 == numBits) {
		uint2 bucketsPacked;
		uint2 offsetsPacked;
		uint predInc = ComputeFusedKeyTotals(fusedKeys, bit, 3, bucketsPacked, 
			offsetsPacked);

		uint4 scanOffsets;
		scanOffsets = MultiScan3(tid, Expand8Uint4To8Uint8(predInc),
			bucketsPacked, offsetsPacked, 0, 0, 0, debug_global);

		SortScatter3_8(scanOffsets, bucketsPacked, offsetsPacked, fusedKeys, 
			packed, tid);
	}
	__syncthreads();
}
	
