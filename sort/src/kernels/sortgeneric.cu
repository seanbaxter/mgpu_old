
// Load 8 fused keys and produce a bucket total and a key offset within
// the thread-bucket. If 4 buckets are being counted, predInc is packed
// with bytes. If 8 buckets are being counted, predInc is packed with
// nibbles. In both cases, offsetsPacked is filled with bytes. This is
// because we'd have to expand the offsets to bytes anyway, so we do it
// here rather than first packing as nibbles.

// returns predInc packed into nibbles (for NUM_BITS=3) or bytes (NUM_BITS = 2).

DEVICE uint LoadFusedKeys8(uint tid, uint bit, uint numBits,
	Values fusedKeys, uint2& bucketsPacked, uint2& offsetsPacked) { 

	// NOTE: this is a stupid construction.
	// tid * VALUES_PER_THREAD is a left shift
	// + index / WARP_SIZE is a right shift
	// the conversion from uint to byte for shared mem LDS is a left shift.
	// At least one of these should be eliminated.
	volatile uint* threadData = scattergather_shared + 
		StridedThreadOrder(tid * VALUES_PER_THREAD);

	uint predInc = 0;
	const int BitsPerValue = (2 == numBits) ? 8 : 4;
		
	#pragma unroll
	for(int v = 0; v < 8; ++v) {
		fusedKeys[v] = threadData[v];
		uint bucket;

		// Note: if this is the first pass for a fused key sort, there is likely
		// optimization potential for exploiting knowledge that bit = 0.
		bucket = bfe(fusedKeys[v], bit, numBits);
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

// Same as SortScatter2, but works on indices that are packed into WORDs. This
// function uses only a single shift to adjust index for bank-conflict
// resolution even when processing two indices. Bitshifts are half-speed
// operations on Fermi, so eliminating one per pair of indices saves 16 cycles
// when sorting 8 values with two passes.

// CONSIDER: pre-multiplying all counters by 4 (shl 2) to avoid having to shift
// by 2 to get dynamic indexing into shared mem for scatter. This is only
// feasible when the stream length is small so as not to overflow the
// byte-packing.

// indexPacked holds the key0 index in the low ushort and the key1 index in the 
// high ushort.


#include "sortscan1.cu"
#include "sortscan2.cu"
#include "sortscan3.cu"

// Read fused keys from shared memory, scan, and scatter the fused keys into
// strided shared memory. 

DEVICE void SortAndScatter(uint tid, uint bit, uint numBits,
	uint numTransBuckets, volatile uint* compressed, 
	volatile uint* uncompressed, bool fusedScatter, uint* debug_global) {

	Values fusedKeys;

	uint packed[4];
	if(1 == numBits) {
		volatile uint* threadData = scattergather_shared + 
			StridedThreadOrder(tid * VALUES_PER_THREAD);
		
		#pragma unroll
		for(int v = 0; v < 8; ++v)
			fusedKeys[v] = threadData[v];
		
		SortScatter1(tid, fusedKeys, bit, packed, numTransBuckets, compressed,
			uncompressed, debug_global);

	} else if(2 == numBits) {
		uint2 bucketsPacked;
		uint2 offsetsPacked;
		uint predInc = LoadFusedKeys8(tid, bit, 2, fusedKeys, bucketsPacked,
			offsetsPacked);

		uint2 scanOffsets;
		scanOffsets = MultiScan2(tid, predInc, numTransBuckets, compressed,
			uncompressed, debug_global);

		SortScatter2_8(scanOffsets, bucketsPacked, offsetsPacked, fusedKeys,
			packed, tid);

	} else if(3 == numBits) {
		uint2 bucketsPacked;
		uint2 offsetsPacked;
		uint predInc = LoadFusedKeys8(tid, bit, 3, fusedKeys, bucketsPacked,
			offsetsPacked);

		uint4 scanOffsets;
		scanOffsets = MultiScan3(tid, Expand8Uint4To8Uint8(predInc),
			bucketsPacked, offsetsPacked, numTransBuckets, compressed, 
			uncompressed, debug_global);

		SortScatter3_8(scanOffsets, bucketsPacked, offsetsPacked, fusedKeys, 
			packed, tid);
	}


	// The sorted values are in thread-order. It is therefore dangerous to
	// scatter without padding due to the high probability of deep bank
	// conflicts when the keys are partially sorted. Eg without padding, on
	// cycle 0, tid 0 writes to 0, tid 1 writes to 8, tid 2 writes to 16, tid 3
	// writes to 24, tid 4 writes to 0, tid 5 writes to 8, etc, producing 8-way
	// bank conflicts for fully sorted keys.

	if(fusedScatter) {
		// Scatter to the original source data index, but with stride, because 
		// we're going from thread order to strided order.
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v) {
			// Pre-multiply to accelerate scatter.
			if(0 == (1 & v)) packed[v / 2]<<= 2;

			// sourceIndex has already been strided and pre-multiplied.
			uint sourceIndex = 0x00ffffff & fusedKeys[v];

			// Store the pre-multiplied scatter indices.
			uint destIndex = (1 & v) ? (packed[v / 2]>> 16) : 
				(0x0000ffff & packed[v / 2]);
			StoreShifted(scattergather_shared, sourceIndex, destIndex);
		}
	} else {
		// Unpack the scatter indices.
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD / 2; ++v) {
			//	indexPacked += (0xffe0ffe0 & scatter)>> 5;
			uint scatter = packed[v];
			scatter = shr_add(0xffe0ffe0 & scatter, 5, scatter);
			scatter<<= 2;			// mul by 4 to convert from int to byte
			uint low = 0x0000ffff & scatter;
			uint high = scatter>> 16;

			StoreShifted(scattergather_shared, low, fusedKeys[2 * v]);
			StoreShifted(scattergather_shared, high, fusedKeys[2 * v + 1]);
		}
	}
	__syncthreads();
}
	
