

////////////////////////////////////////////////////////////////////////////////
// PHASE THREE
// Phase one built an exclusive scan for the first row of bucket counts within
// each warp of the histogram. Phase two computed bucket totals and updated 
// countScan_global to be globally relevent.

// Input:
// bucketCount_global is the output of the count kernel.
//		Bucket counts for each sort block packed into 16bit uints.
// rangePairs_global is the start and end interval for each warp in this pass.
//		These values always refers to warp's worth of data, so multiply by 32.
// countScan_global is the exclusive scan for the starting point for each
//		bucket. Input for each histogram block is NUM_BUCKETS, totalling 
//		NumSMs * NUM_BUCKETS.
// columnScan_global is the scan of bucket counts for totals within each warp of
//		this block. This is required as it gives each warp in the phase 3 
//		histogram kernel a starting scatter offset for each bucket. This data is
//		not read by the phase 2 kernel. However, the countScan_globals are
//		modified by the phase 2 kernel and added to columnScan_global in
//		phase 3.
// Output:
// bucketCodes_global

// Each warp of the sort kernel requires a transaction list. While we could 
// choose to include only NUM_SORT_WARPS transaction lists, including WARP_SIZE
// lists makes list expansion in the sort kernel more efficient, as all threads
// in the warp can expand a list, rather than just 8 of them 
// (for NUM_SORT_THREADS = 256). Transaction list expansion occurs in parallel
// with single-warp sequential multi-scan for a 3-bit sort
// (or 2-bit sort when NUM_BITS=4), so the transaction list must be kept as 
// short as possible.

#pragma once

#include "common.cu"

// Scan from right into left. This can be function is parameterized to be used
// either to compute scatters (the same bucket over multiple sort blocks) or
// gathers (different buckets within a sort block) dependending on pred and
// stride.
DEVICE uint HistParallelScan(uint pred, uint stride, uint x, int loops, 
	volatile uint* scan) {

	uint sum = x;
	scan[0] = x;
	#pragma unroll
	for(int i = 0; i < loops; ++i) {
		uint offset = stride<< i;
		uint y = scan[-offset];
		if(pred >= offset) x += y;
		scan[0] = x;
	}
	return x - sum;
}


////////////////////////////////////////////////////////////////////////////////
// Given a single count per thread and a scatter for each bucket in the first 
// sort block (i.e. when lane < NUM_BUCKETS), compute scatter and gather indices
// and write to global memory. This function is called twice from the hist3
// kernel. Return the next scatter offset for the lane (when lane < NUM_BUCKETS).
/*
template<int NumBits>
DEVICE2 uint ComputeGatherStruct(uint* bucketCodes_global, int numSortBlocks,
	uint tid, uint warp, uint lane, uint bucket, uint scatter, uint count,
	uint earlyExitMask) {

	volatile uint* scatterScan = scatterScan_shared3 + tid;

	volatile uint* gatherScan = gatherScan_shared3 + tid;

	// Compute the scatter for each bucket in each sort block.
	// scatter is defined only for lane < NUM_BUCKETS. The scan will pull this
	// global scatter offset to all sort blocks to the right. HistParallelScan
	// is an exclusive scan, so to prevent it from returning 0 for the first
	// sort block, add the scatter offset back in.

	// Note that the high bit of the first bucket count for each sort group is
	// the sort detect flag. Clear this before doing a parallel scan.

	uint sortedFlag = earlyExitMask & count;
	count &= 0x00007fff;
	scatter += HistParallelScan(lane, NUM_BUCKETS, scatter + count, 
		5 - NUM_BITS, scatterScan);

	uint nextScatter = 0;
	if(lane < NUM_BUCKETS)
		nextScatter = scatterScan[NUM_BUCKETS * (numSortBlocks - 1)];

	// Store the exclusive scatter offsets back into scatter
	scatterScan[0] = scatter;

	// Compute the gather from the counts
	uint gather = HistParallelScan(bucket, 1, count, NUM_BITS, gatherScan);

	// Store the gather indices back in gatherScan, along with sortedFlag.
	gatherScan[0] = gather + sortedFlag;

	// Loop over each sort block and serialize
	#pragma unroll
	for(int i = 0; i < numSortBlocks; ++i) {
		int offset = NUM_BUCKETS * i;
		if(lane < NUM_BUCKETS) {
			 // Subtract gather index from scatter index. Multiply by 4 to avoid
			// a SHL in the sort kernel.
			gather = gatherScan[offset];
			sortedFlag = gather>> 15;
			gather &= 0x00007fff;

			// Make the sorted flag the least significant bit in the first value
			// of the pre-multiplied scatter indices.
			bucketCodes_global[lane] =
				4 * (scatterScan[offset] - gather) + sortedFlag;
		}
		bucketCodes_global += TRANS_STRUCT_SIZE;
	}
	return nextScatter;
}
*/

template<int NumBits>
DEVICE2 uint ComputeGatherStruct(uint* bucketCodes_global, int numSortBlocks,
	uint tid, uint warp, uint lane, uint bucket, uint scatter, uint count,
	uint earlyExitMask) {

template<int NumThreads, int NumBits>
DEVICE2 void HistogramFunc3(const uint* bucketCount_global, int rangeQuot,
	int rangeRem, int segSize, int count, const uint* countScan_global, 
	const uint* columnScan_global, uint* bucketCodes_global, 
	int supportEarlyExit) {

	const int NumWarps = NumThreads / WARP_SIZE;
	const int NumDigits = 1<< NumBits;
	const int NumChannels = NumDigits / 2;

	const int TransStructSize = NumDigits;
	const int BlocksPerWarp = MIN(NUM_COUNT_WARPS, WarpStride / NumChannels);

	const int CountersSize = MIN(NumDigits / 2, WARP_SIZE);

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

//	uint channel = (NumChannels - 1) & tid;
//	uint sortBlock = lane / NumChannels;
	
	// uint2 range = rangePairs_global[NUM_WARPS * block + warp];
	int2 range = ComputeTaskRange(NumWarps * block + warp, rangeQuot, rangeRem,
		segSize, count);

	__shared__ volatile uint scatterScan_shared[NumDigits];
	__shared__ volatile uint columnScan_shared[NumDigits * NumWarps];
	__shared__ volatile uint bucketCodes_shared[
		TransStructSize * BlocksPerWarp * NumWarps];
	
	// Find the starting scatter offsets for the first sort block in for each
	// warp in this histogram pass. Phase one has already computed the offsets
	// for the first sort block in each warp within this histogram block and 
	// cached the values in columnScan_global. We take those values and add the
	// global bucket scatter offsets from countScan_global.
	if(tid < NumDigits)
		scatterScan_shared3[tid] = countScan_global[NumDigits * block + tid];

	// columnScan_global indicates where each column of the packed block
	// offsets should start. Add these with countScan_global.
	for(int i = tid; tid < NumDigits * NumWarps; tid += NumThreads)
		columnScan_shared[i] = 
			columnScan_global[NumDigits * NumWarps * block + i];

	__syncthreads();

	uint scatter0 = 0, scatter1 = 0, scatter2 = 0, scatter3 = 0;
	if(NumBits <= 5) {
		if(lane < NumChannels) {
		}
	} else if(NumBits <= 7) {
		// scatter0 = digit lane.
		// scatter1 = digit WARP_SIZE + lane.
		volatile uint* scan = columnScan_shared + NumDigits * warp + lane;
		scatter0 = scan[0] + scatterScan_shared[lane];
		scatter1 = scan[WARP_SIZE] + scatterScan_shared[WARP_SIZE + lane];
		if(7 == NumBits) {
			scatter2 = scan[2 * WARP_SIZE] + 
				scatterScan_shared[2 * WARP_SIZE + lane];
			scatter3 = scan[3 * WARP_SIZE] + 
				scatterScan_shared[3 * WARP_SIZE + lane];
		}
	}


	////////////////////////////////////////////////////////////////////////////
	// Loop over every warp's worth of data and accumulate global offsets.

	// Advance bucketCodes_global to the next set of digit counts for this warp.
	bucketCodes_global += TransStructSize * BlocksPerWarp * range.x;

	volatile uint* bucketCodes_warp = bucketCodes_shared + 
		TransStructSize * BlocksPerWarp * warp;

	uint current = range.x;
	while(current < range.y) {
		uint offset = CountersSize * current + lane;
		uint packed0 = bucketCount_global[offset];

		if(NumBits <= 5) {
			// Run an exclusive scan over multiple columns (sort blocks) in this
			// warp. We only have multiple columns for NumBits <= 5.
			uint x = 0;
			bucketCount_global[lane] = packed0;
			if(lane < NumChannels) {
				#pragma unroll
				for(int i = 0; i < BlocksPerWarp; ++i) {
					uint y = bucketCodes_warp[lane + NumChannels * i];
					bucketCodes_warp[lane + NumChannels * i] = x;
					x += y;
				}
			}
			scatter = bucketCount_global[lane] - packed0;			
		} else if(6 == NumBits) {

		}

		// Extract the packed counts to low and high parts.
		uint count0 = 0x0000ffff & packed0;
		uint count1 = packed>> 16;
		uint count2, count3;

		if(7 == NumBits) {
			uint packed1 = bucketCount_global[WARP_SIZE + offset];
			count2 = 0x0000ffff & packed1;
			count3 = packed1>> 16;
		}

		// Re-order 








		++current;
	}






	// Each thread loads a low and high counter. The block sort managed by each
	// thread changes every NUM_CHANNELS threads.
	volatile uint* countWarp = columnScan_shared3 + 2 * WARP_SIZE * warp;
	volatile uint* countBlock = countWarp + NUM_BUCKETS * sortBlock + channel;

	// We want to exchange the scatter and count values to reduce bank conflicts
	// and simplify code.
	countBlock[0] = scatter1;
	countBlock[NUM_CHANNELS] = scatter2;
	uint scatter = countWarp[lane];

	bucketCodes_global += 
		TRANS_STRUCT_SIZE * NUM_SORT_BLOCKS_PER_WARP * range.x;

	uint current = range.x;
	while(current < range.y) {

		uint packed = bucketCount_global[WARP_SIZE * current + lane];
		uint count1 = 0x0000ffff & packed;
		uint count2 = packed>> 16;

		// Exchange the counters to match the scatter indices.
		countBlock[0] = count1;
		countBlock[NUM_CHANNELS] = count2;
		count1 = countWarp[lane];
	
#ifdef INCLUDE_TRANSACTION_LIST
		// TODO: Get the number of sort blocks for each pass.
		scatter = ComputeTransactionList(bucketCodes_global, 
			NUM_HALF1_SORT_BLOCKS, tid, warp, lane, bucket, scatter, 
			count1, earlyExitShift);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF1_SORT_BLOCKS;

#if NUM_HALF2_SORT_BLOCKS > 0
		count2 = countWarp[WARP_SIZE + lane];
		scatter = ComputeTransactionList(bucketCodes_global, 
			NUM_HALF2_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count2, earlyExitShift);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF2_SORT_BLOCKS;
#endif
#else // !defined(INCLUDE_TRANSACTION_LIST)
		scatter = ComputeGatherStruct(bucketCodes_global,
			NUM_HALF1_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count1, earlyExitMask);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF1_SORT_BLOCKS;
		
#if NUM_HALF2_SORT_BLOCKS > 0
		count2 = countWarp[WARP_SIZE + lane];
		scatter = ComputeGatherStruct(bucketCodes_global, 
			NUM_HALF2_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count2, earlyExitMask);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF2_SORT_BLOCKS;
#endif

#endif
		++current;
	}


















	// The early exit flag comes in the most significant bit of the low short
	// of the first value in each sort block's radix digit frequency array.
#ifdef INCLUDE_TRANSACTION_LIST
	uint earlyExitShift = supportEarlyExit ? 15 : 16;
#else
	uint earlyExitMask = supportEarlyExit ? 0x8000 : 0x0000;
#endif

	// Zero out the unused part of the bucket codes structure
	bucketCodes_shared3[tid] = 0;
	bucketCodes_shared3[NUM_THREADS + tid] = 0;
	bucketCodes_shared3[2 * NUM_THREADS + tid] = 0;

	
	if(tid < NUM_BUCKETS * NUM_WARPS)
		columnScan_shared3[tid] = 
			columnScan_global[NUM_BUCKETS * NUM_WARPS * block + tid];

	__syncthreads();
	
	// Each thread reads packed offsets, consisting of a low count and a high
	// count. If lane < NUM_CHANNELS (NUM_BUCKETS / 2), then this thread will
	// encounter the first counters for the low and high bucket in this warp,
	// and they are responsible for setting the scatter offsets.
	uint scatter1 = 0;
	uint scatter2 = 0;
	if(lane < NUM_CHANNELS) {
		scatter1 = 
			columnScan_shared3[warp * NUM_CHANNELS + lane] +
			scatterScan_shared3[lane];
		scatter2 = 
			columnScan_shared3[(NUM_WARPS + warp) * NUM_CHANNELS + lane] +
			scatterScan_shared3[NUM_CHANNELS + lane];
	}

	__syncthreads();

	// Each thread loads a low and high counter. The block sort managed by each
	// thread changes every NUM_CHANNELS threads.
	volatile uint* countWarp = columnScan_shared3 + 2 * WARP_SIZE * warp;
	volatile uint* countBlock = countWarp + NUM_BUCKETS * sortBlock + channel;

	// We want to exchange the scatter and count values to reduce bank conflicts
	// and simplify code.
	countBlock[0] = scatter1;
	countBlock[NUM_CHANNELS] = scatter2;
	uint scatter = countWarp[lane];

	bucketCodes_global += 
		TRANS_STRUCT_SIZE * NUM_SORT_BLOCKS_PER_WARP * range.x;

	uint current = range.x;
	while(current < range.y) {

		uint packed = bucketCount_global[WARP_SIZE * current + lane];
		uint count1 = 0x0000ffff & packed;
		uint count2 = packed>> 16;

		// Exchange the counters to match the scatter indices.
		countBlock[0] = count1;
		countBlock[NUM_CHANNELS] = count2;
		count1 = countWarp[lane];
	
#ifdef INCLUDE_TRANSACTION_LIST
		// TODO: Get the number of sort blocks for each pass.
		scatter = ComputeTransactionList(bucketCodes_global, 
			NUM_HALF1_SORT_BLOCKS, tid, warp, lane, bucket, scatter, 
			count1, earlyExitShift);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF1_SORT_BLOCKS;

#if NUM_HALF2_SORT_BLOCKS > 0
		count2 = countWarp[WARP_SIZE + lane];
		scatter = ComputeTransactionList(bucketCodes_global, 
			NUM_HALF2_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count2, earlyExitShift);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF2_SORT_BLOCKS;
#endif
#else // !defined(INCLUDE_TRANSACTION_LIST)
		scatter = ComputeGatherStruct(bucketCodes_global,
			NUM_HALF1_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count1, earlyExitMask);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF1_SORT_BLOCKS;
		
#if NUM_HALF2_SORT_BLOCKS > 0
		count2 = countWarp[WARP_SIZE + lane];
		scatter = ComputeGatherStruct(bucketCodes_global, 
			NUM_HALF2_SORT_BLOCKS, tid, warp, lane, bucket, scatter,
			count2, earlyExitMask);
		bucketCodes_global += TRANS_STRUCT_SIZE * NUM_HALF2_SORT_BLOCKS;
#endif

#endif
		++current;
	}
}




/*
////////////////////////////////////////////////////////////////////////////////
// Given a single count per thread and a scatter for each bucket in the first
// sort block (i.e. when lane < NUM_BUCKETS), compute the transaction list 
// structure. This function is called twice from the hist3 kernel.
// Return the next scatter offset for the lane (when lane < NUM_BUCKETS).

DEVICE uint ComputeTransactionList(uint* bucketCodes_global, int numSortBlocks,
	uint tid, uint warp, uint lane, uint bucket, uint scatter, uint count,
	uint earlyExitShift) {

	volatile uint* countWarp = columnScan_shared3 + 2 * WARP_SIZE * warp;
	volatile uint* countScan = countWarp + lane;

	volatile uint* scatterScan = scatterScan_shared3 + tid;

	volatile uint* gatherScan = gatherScan_shared3 + tid;

	volatile uint* listScan = listScan_shared3 + tid;

	volatile uint* transWarp = transOffset_shared3 + WARP_SIZE * warp;
	volatile uint* transScan = transOffset_shared3 + tid;

	volatile uint* bucketCodesWarp = bucketCodes_shared3 + 3 * WARP_SIZE * warp;
	volatile uint* bucketCodesScan = bucketCodesWarp + lane;

	uint sortedFlag = 0;// count>> earlyExitShift;
	count &= 0x00007fff;

	// Find the next non-empty bucket. Keep in register until it's needed again.
	uint next = GetNextOccupiedBucket(bucket, count, NUM_BITS, countScan);

	// Compute the scatter for each bucket in each sort block.
	scatter += HistParallelScan(lane, NUM_BUCKETS, scatter + count, 
		5 - NUM_BITS, scatterScan);

	uint nextScatter = 0;
	if(lane < NUM_BUCKETS)
		nextScatter = scatterScan[NUM_BUCKETS * (numSortBlocks - 1)];


	// Bake in the scatter flag for the first bucket in each sort block.
	// Store the exclusive scatter offsets back into scatterScan memory.
	scatterScan[0] = scatter + sortedFlag;

	// Compute the transaction count per bucket and fuse with the bucket count
	// and scan.
	uint transCount = NumBucketTransactions(scatter, count);
	uint fused = bfi(count, transCount, 16, 16);
	fused = HistParallelScan(bucket, 1, fused, NUM_BITS, countScan);

	uint gather = 0x0000ffff & fused;
	uint transOffset = fused>> 16;

	// Grab the transaction offset (inclusive) from the last bucket in the 
	// sort block.
	uint transTotal = countWarp[lane | (NUM_BUCKETS - 1)]>> 16;
	transScan[0] = transOffset;

	// Store the total number of transactions in the gather[0] offset
	if(!bucket) gather = transTotal;

	// Pack these three terms together to describe each bucket in the
	// transaction struct:
	// 14:0 - gather index into block shared memory.
	// 24:15 - number of transactions for data in this bucket.
	// 31:25 - index of the next non-zero bucket. establishes a linked list.
	uint bucketCode = bfi(gather, transCount, 15, 10);
	bucketCode = bfi(bucketCode, next, 25, 7);
	gatherScan[0] = bucketCode;

	// Map each bucket to a list and cache the counters in listScan.
	uint list = MapBucketToList(transOffset, transCount, transTotal);
	listScan[0] = list;

	// Loop over each sort block.
	#pragma unroll	
	for(int sortBlock = 0; sortBlock < numSortBlocks; ++sortBlock) {
		// The list indices from MapBucketToList need to be mapped to shared 
		// memory and scanned to establish the first bucket for each list.
		int sortIndex = NUM_BUCKETS * sortBlock;

		// grab the transaction total out of countScan
	#if 5 != NUM_BITS
		transTotal = countWarp[sortIndex + NUM_BUCKETS - 1]>> 16;
	#endif

		// Use bucketCodes_shared as scratch memory to find the starting bucket
		// for each transaction list.
		bucketCodesScan[0] = 0;
		list = 0xffffffff;
		if(lane < NUM_BUCKETS) list = listScan[sortIndex];
		if(0xffffffff != list) bucketCodesWarp[list] = lane;

		// Drag bucket lists from left to right.
		uint firstBucket = bucketCodesScan[0];
		#pragma unroll
		for(int i = 0; i < LOG_NUM_TRANSACTION_LISTS; ++i) {
			uint offset = 1<< i;
			uint y = bucketCodesScan[-offset];
			uint y2 = max(firstBucket, y);
			if(offset <= lane) firstBucket = y2;
			if(i < LOG_NUM_TRANSACTION_LISTS - 1) 
				bucketCodesScan[0] = firstBucket;			
		}

		// Compute the number of transactions in this lane's list.
		uint numListTrans = NumListTransactions(lane, transTotal);
		uint listTransOffset = ListTransactionOffset(lane, transTotal);

		// Look up the transaction offset for the first transaction in this
		// list.
		uint firstBucketOffset = listTransOffset -
			transWarp[sortIndex + firstBucket];

		// Pack these three terms together to establish each list:
		// 6:0 - index of the bucket at which to begin global stores.
		// 16:7 - first transaction of the first bucket. After this, the
		//		warp traverses the buckets like a linked list
		// 31:17 - total number of transactions for this list to process.
		uint listCode = bfi(firstBucket, firstBucketOffset, 7, 10);
		listCode = bfi(listCode, numListTrans, 17, 15);

		// Put all the values into bucketCodes_shared3 and write to global
		// memory.
		if(lane < NUM_BUCKETS) {
			bucketCodesScan[0] = scatterScan[sortIndex];
			bucketCodesScan[NUM_BUCKETS] = gatherScan[sortIndex];
		}
		bucketCodesScan[2 * NUM_BUCKETS] = listCode;

		#pragma unroll
		for(int j = 0; j < TRANS_STRUCT_SIZE; j += WARP_SIZE)
			bucketCodes_global[j + lane] = bucketCodesScan[j];

		bucketCodes_global += TRANS_STRUCT_SIZE;
	}
	
	return nextScatter;
}*/