#pragma once

#include "sortcommon.cu"
#include "sortscan1.cu"
#include "sortscan2.cu"
#include "sortscan3.cu"
#include "sortscan4.cu"

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
			predInc = 1ull<< shift;
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
			prevPredInc &= 0x0f;
			offsetsPacked.y = prevPredInc;
			bucketsPacked = bfi(bucketsPacked, digit, 4 * v, 4);
		} else {
			offsetsPacked.y = bfi(offsetsPacked.y, prevPredInc, 8 * (v - 4),
				BitsPerValue);
			bucketsPacked = bfi(bucketsPacked, digit, 4 * v, 4);
		}

		if(v) predInc += 1ull<< shift;
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
		uint4 predInc4 = make_uint4(
			predIncLow.x, predIncLow.y, predIncHigh.x, predIncHigh.y);
			
		uint4 offsetsLow, offsetsHigh;
#ifdef SCAN4_1WARP
		MultiScan4_1Warp(tid, predInc4, numThreads, bucketsPacked, 
			offsetsLow, offsetsHigh, scratch_shared, debug_global);
#endif
		SortScatter4_8(offsetsLow, offsetsHigh, bucketsPacked, predInc4,
			offsetsPacked, packed);
	}
}



////////////////////////////////////////////////////////////////////////////////
// SortLocal

// Given keys in thread order (fusedKeys) or keys in shared memory in strided
// order (scattergather_shared), sort between 1 and 7 key bits and store into
// shared memory.

template<int NumThreads, int NumBits>
DEVICE2 void SortLocal(uint tid, Values fusedKeys, uint bit, bool loadFromArray,
	uint* scattergather_shared, uint* scratch_shared, uint* debug_global) {

	if(1 == NumBits) 
		SortAndScatter(tid, fusedKeys, bit, 1, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(2 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(3 == NumBits)
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray,
			false, scattergather_shared, scratch_shared, debug_global);
	else if(4 == NumBits) {
		/*SortAndScatter(tid, fusedKeys, bit, 4, NumThreads,
			!LoadFromTexture, false, scattergather_shared, scratch_shared, 
			debug_global_out);*/
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(5 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(6 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 3, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 3, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	} else if(7 == NumBits) {
		SortAndScatter(tid, fusedKeys, bit, 2, NumThreads, loadFromArray, 
			true, scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 2, 2, NumThreads, true, true,
			scattergather_shared, scratch_shared, debug_global);
		SortAndScatter(tid, fusedKeys, bit + 4, 3, NumThreads, true, false,
			scattergather_shared, scratch_shared, debug_global);
	}
}


////////////////////////////////////////////////////////////////////////////////
// LoadKeysGlobal

template<int NumThreads, int NumBits, bool LoadFromTexture>
DEVICE2 void LoadKeysGlobal(uint tid, uint block, const uint* keys_global_in,
	uint texBlock, uint bit, uint* scattergather_shared, bool useFusedKey, 
	Values keys, Values fusedKeys) {

	const int NumValues = VALUES_PER_THREAD * NumThreads;

	////////////////////////////////////////////////////////////////////////////
	// LOAD KEYS, CREATE FUSED KEYS, AND REINDEX INTO THREAD ORDER

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	// Load the keys and, if sorting values, create fused keys. Store into 
	// shared mem with a WARP_SIZE + 1 stride between warp rows, so that loads
	// into thread order occur without bank conflicts.
	if(LoadFromTexture) {
		// Load keys from a texture. The texture sampler serves as an 
		// asynchronous independent subsystem. It helps transpose data from
		// strided to thread order without involving the shader units.

		uint keysOffset = NumValues * texBlock + VALUES_PER_THREAD * tid;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
			uint4 k = tex1Dfetch(keys_texture_in, keysOffset / 4 + i);
			keys[4 * i + 0] = k.x;
			keys[4 * i + 1] = k.y;
			keys[4 * i + 2] = k.z;
			keys[4 * i + 3] = k.w;
		}

		if(!useFusedKey)
			// Sort only keys.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i)
				fusedKeys[i] = keys[i];
		else
			// Sort key-value tuples.
			BuildFusedKeysThreadOrder(tid, keys, bit, NumBits, fusedKeys,
				false);

	} else {
		// Load keys from global memory. This requires using shared memory to 
		// transpose data from strided to thread order.

		LoadWarpValues(keys_global_in, warp, lane, block, keys);

		if(!useFusedKey)
			// Sort only keys.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i)
				fusedKeys[i] = keys[i];
		else
			// Sort key-value tuples.
			BuildFusedKeysWarpOrder(warp, lane, keys, bit, NumBits, fusedKeys,
				false);

		// Store the keys or fused keys into shared memory for the
		// strided->thread order transpose.
		ScatterWarpOrder(warp, lane, true, fusedKeys, scattergather_shared);
	}
}


////////////////////////////////////////////////////////////////////////////////
// LoadAndSortLocal

template<int NumThreads, int NumBits, bool LoadFromTexture>
DEVICE2 void LoadAndSortLocal(uint tid, uint block, const uint* keys_global_in, 
	uint texBlock, uint bit, uint* debug_global_out, uint* scattergather_shared,
	uint* scratch_shared, bool useFusedKey, Values keys, Values fusedKeys) {

	LoadKeysGlobal<NumThreads, NumBits, LoadFromTexture>(tid, block,
		keys_global_in, texBlock, bit, scattergather_shared, useFusedKey,
		keys, fusedKeys);

	uint scanBitOffset = useFusedKey ? 24 : bit;

	SortLocal<NumThreads, NumBits>(tid, fusedKeys, scanBitOffset,
		!LoadFromTexture, scattergather_shared, scratch_shared, 
		debug_global_out);
}
	