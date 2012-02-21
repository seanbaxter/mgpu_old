#pragma once

#include "common.cu"

////////////////////////////////////////////////////////////////////////////////
// STORE IN-PLACE

DEVICE void WriteKeysInplace(uint block, uint tid, uint* sorted_global,
	const Values keys) {

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		sorted_global[NUM_VALUES * block + v * NUM_THREADS + tid] = keys[v];
}



////////////////////////////////////////////////////////////////////////////////
// STORE SIMPLE (scatter for global sort, but don't perform coalescing
// optimizations)

// StoreToGlobal takes two offsets. This is to better control the associativity
// of the adds. Much better code is generated if the adds left-associate.
// Passing a single pointer offset would essentially right-associate the two 
// adds, causing nvcc to generate suboptimal code, and failing to exploit VADD
// for 3-parameter addition.

DEVICE void ScatterKeysSimple(uint tid, uint* sorted_global, uint bitOffset, 
	uint numBits, const uint* compressed, const Values keys) {

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint key = keys[v];
		uint bucket = bfe(key, bitOffset, numBits);

		// Get the gather offset
		uint index = 4 * (NUM_THREADS * v + tid);

		// uint offset = index - compressedList_shared[numBuckets + bucket];
		// uint scatter = compressedList_shared[bucket] + offset;

		// We've already subtracted the gather offset from the scatter offset.
		// This reduces the size of the scatter list and saves some dynamic
		// LDSs.
		uint scatter = compressed[bucket];
		
		// Add the index and scatter integers here in the pointer arithmetic. 
		// This coaxes nvcc to generate a VADD instruction to add together all
		// three integers (the pointer, index, and scatter) in a single 
		// instruction, saving numerous slots!
		StoreToGlobal2(sorted_global, scatter, index, key);
	}
}

DEVICE void ScatterPairSimple(uint tid, uint* first_global, uint* second_global,
	uint bitOffset, uint numBits, const uint* compressed, 
	const Values first, const Values second) {

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint bucket = bfe(first[v], bitOffset, numBits);

		// Get the gather offset
		uint index = 4 * (NUM_THREADS * v + tid);

		// uint offset = index - compressedList_shared[numBuckets + bucket];
		// uint scatter = compressedList_shared[bucket] + offset;

		// We've already subtracting the gather offset from the scatter offset.
		// This reduces the size of the scatter list and saves some dynamic
		// LDSs.
		uint scatter = compressed[bucket];

		// Add the index and scatter integers here in the pointer arithmetic. 
		// This coaxes nvcc to generate a VADD instruction to add together all
		// three integers (the pointer, index, and scatter) in a single 
		// instruction, saving numerous slots!
		StoreToGlobal2(first_global, scatter, index, first[v]);
		StoreToGlobal2(second_global, scatter, index, second[v]);
	}
}

DEVICE void MultiScatterSimple(uint tid, uint* keys_global, uint bitOffset,
	uint numBits, const uint* compressed, const Values keys,
	Values globalScatter) {

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint bucket = bfe(keys[v], bitOffset, numBits);

		// Get the gather offset
		uint index = 4 * (NUM_THREADS * v + tid);

		// uint offset = index - compressedList_shared[numBuckets + bucket];
		// uint scatter = compressedList_shared[bucket] + offset;

		// We've already subtracting the gather offset from the scatter offset.
		// This reduces the size of the scatter list and saves some dynamic 
		// LDSs.
		uint scatter = compressed[bucket];
		globalScatter[v] = scatter + index;

		StoreToGlobal(keys_global, globalScatter[v], keys[v]);
	}
}

DEVICE void GlobalGatherScatter(uint tid, uint block, uint numThreads,
	const uint* source, uint* dest, uint* scattergather_shared, 
	const Values gather, const Values globalScatter) {

	__syncthreads();

	Values values;
	LoadBlockValues(source, tid, block, values);
	ScatterBlockOrder(tid, false, numThreads, values, scattergather_shared);
	__syncthreads();

	GatherFromIndex(gather, true, values, scattergather_shared);

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		StoreToGlobal(dest, globalScatter[v], values[v]);
}


////////////////////////////////////////////////////////////////////////////////
// STORE TRANSACTION LIST

// 1) uint histogramOffsets[NUM_BUCKETS]
// 2) uint bucketCodes[NUM_BUCKETS]
// bits 14:0 (0 to 16383) offset of the start of the bucket in the sorted
//		array. this is an exclusive scan. the offset of the first bucket is set
//		to the total number of defined keys for the group.  The count for each
//		bucket can be computed by subtracting the bucket's offset from the next
//		bucket's offset.

// bits 24:15 (0 to 1023) hold the number of transactions to be performed on
//		this bucket.

// bits 31:25 (0 to 127) index of the next bucket with a non-zero number of
//		requests. This creates a linked list of requests for each warp to 
//		traverse.

// 3) uint warpCodes[NUM_THREADS / WARP_SIZE]
// bits 6:0 (0 to 127) index of the bucket at which to begin writes.
// bits 16:7 (0 to 1023) first transaction of the first bucket. After this
//		the warp traverses the buckets like a linked list.
// bits 31:17 (0 to 31) number of transactions to process.

// This struct is aligned to a multiple of the segment size (WARP_SIZE).
// Typical usage: NUM_THREADS = 1024, VALUES_PER_THREAD = 8, NUM_BUCKETS = 64:
// NUM_THREADS / WARP_SIZE = 32. Each struct is 160 bytes, or 5 transactions.
/*
DEVICE uint2 GetTransactionInterval(uint scatter, uint count, uint request) {
	uint start = ~(WARP_SIZE - 1) & (scatter + request * WARP_SIZE);
	uint end = min(start + WARP_SIZE, scatter + count);
	start = max(start, scatter);
	return make_uint2(start, end);
}


// ExpandScatterList writes a count per transaction list and two words (scatter
// and gather codes) per transaction for each list. Because we know the maximum
// length of each list, we can store the scatter and gather codes 
// MaxTransactions apart in shared memory.

DEVICE void ExpandScatterList(uint lane, uint numBuckets,
	volatile uint* compressed, volatile uint* uncompressed,
	uint* debug_global_out) {

	const uint MaxTransactions = MAX_TRANS(NUM_VALUES, numBuckets);

	// The total number of transactions is stored in the gather offset for 
	// bucket 0. Have lane 0 move this into totalTransactions_shared.
	uint firstBucket = compressed[numBuckets];
	uint totalTrans = 0x7fff & firstBucket;
	firstBucket &= ~0x7fff;		// clear the gather index to 0
	if(!lane) compressed[numBuckets] = firstBucket;

	uint n = totalTrans / NUM_WARPS;
	uint m = (NUM_WARPS - 1) & totalTrans;
	uint warpTransOffset = lane * n + min(lane, m);
	uint warpTransCount = n + (lane < m);
	uint packedCount = bfi(warpTransCount, warpTransOffset, 16, 16);
	
	if(lane < NUM_WARPS) 
		uncompressed[2 * MaxTransactions + lane] = packedCount;
	
	// Each warp its transaction count, first bucket, and bucket offset.
	uint warpCode = compressed[2 * numBuckets + lane];
	
	// Index of the bucket at which to begin writes
	uint curBucket = 0x7f & warpCode;

	// First transaction of the first bucket. After this the warp traverses
	// the buckets like a linked list.
	uint curTrans = bfe(warpCode, 7, 10);

	// Number of transactions for this warp to process.
	uint transCount = warpCode>> 17;

	for(int i = 0; i < transCount; ++i) {
		uint bucketCode = compressed[numBuckets + curBucket];

		// Get the gather and scatter indices for this bucket. 
		uint gather = 0x7fff & bucketCode;
		uint end = (curBucket < numBuckets - 1) ? 
			(0x7fff & compressed[numBuckets + 1 + curBucket]) : NUM_VALUES;
		uint scatter = compressed[curBucket];
		
		// GetTransactionInterval returns the start and end offsets in global
		// memory for the scatter.
		uint2 writePair = 
			GetTransactionInterval(scatter, end - gather, curTrans);

		// Gather index for the first value in this transaction.
		uint gather2 = gather + writePair.x - scatter;
		uint count = writePair.y - writePair.x;

		uncompressed[lane + i * WARP_SIZE] = writePair.x;
		uncompressed[MaxTransactions + lane + i * WARP_SIZE] =
			bfi(count, gather2, 16, 16);

		// Move to the next bucket if we processed the last transaction in the
		// bucket.
		uint bucketTrans = bfe(bucketCode, 15, 10);
		++curTrans;
		if(curTrans >= bucketTrans) {
			curBucket = bucketCode>> 25;
			curTrans = 0;
		}
	}
}

*/
/*
// WriteCoalesced is for transaction list support.

DEVICE void WriteCoalesced(uint warp, uint lane, uint numBuckets, bool strided,
	const uint* uncompressed, uint* sorted_global) {

	const uint MaxTransactions = MAX_TRANS(NUM_VALUES, numBuckets);

	uint lane4 = 4 * lane;

	uint packedCount = uncompressed[2 * MaxTransactions + warp];
	uint transOffset = packedCount>> 16;
	uint transCount = 0x0000ffff & packedCount;

	uncompressed += transOffset;

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint scatterCode = uncompressed[i];
		uint gatherCode = uncompressed[MaxTransactions + i];
		
		uint count = 0x0000ffff & gatherCode;

		uint gather, value;
		if(strided) {
			gather = (gatherCode>> 16) + lane;
			gather = shr_add(gather, 5, gather);
			value = scattergather_shared[gather];
		}
		if(!strided) gather = shr_add(gatherCode, 14, lane4);
		if(!strided) value = LoadShifted(scattergather_shared, gather);
		
		if(lane < count) sorted_global[scatterCode] = value;
	}

	for(int i = VALUES_PER_THREAD; i < transCount; ++i) {
		uint scatterCode = uncompressed[i];
		uint gatherCode = uncompressed[MaxTransactions + i];
		
		uint count = 0x0000ffff & gatherCode;

		uint gather, value;
		if(strided) {
			gather = (gatherCode>> 16) + lane;
			gather = shr_add(gather, 5, gather);
			value = scattergather_shared[gather];
		} else {
			gather = shr_add(gatherCode, 14, lane4);
			value = LoadShifted(scattergather_shared, gather);
		}
		if(lane < count) sorted_global[scatterCode] = value;
	}
}


// Write the first set of data from shared memory and pack the first 8 gather
// and adjusted scatter pointers or indices. This avoids additional LDS
// instructions for subsequent write calls. The counts are dropped in favor of
// predicates (bool preds[8]).

DEVICE void WriteCoalescedAndCache(uint warp, uint lane, uint numBuckets,
	bool cache, const uint* uncompressed, uint* sorted_global, Values counts,
	Values gather, Values scatter) {

	const uint MaxTransactions = MAX_TRANS(NUM_VALUES, numBuckets);

	uint lane4 = 4 * lane;

	uint packedCount = uncompressed[2 * MaxTransactions + warp];
	uint transOffset = packedCount>> 16;
	uint transCount = 0x0000ffff & packedCount;

	uncompressed += transOffset;

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		if(cache) {
			
			scatter[i] = uncompressed[i];
		}
			uint gatherCode = uncompressed[MaxTransactions + i];
			counts[i] = 0x0000ffff & gatherCode;
			gather[i] = shr_add(gatherCode, 14, lane4);
		uint value = LoadShifted(scattergather_shared, gather[i]);
		uint* global = (uint*)((char*)(sorted_global) + scatter[i]);

		if(lane < counts[i]) *global = value;
	}

	for(int i = VALUES_PER_THREAD; i < transCount; ++i) {
		uint scatterCode = uncompressed[i];
		uint gatherCode = uncompressed[MaxTransactions + i];
		
		uint count = 0x0000ffff & gatherCode;

		uint gather = shr_add(gatherCode, 14, lane4);
		uint value = LoadShifted(scattergather_shared, gather);
		
		uint* global = (uint*)((char*)(sorted_global) + scatterCode);
		if(lane < count) *global = value;
	}
}

*/