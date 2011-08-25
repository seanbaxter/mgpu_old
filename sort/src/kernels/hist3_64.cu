
////////////////////////////////////////////////////////////////////////////////
// 6 == NUM_BITS HISTOGRAM_FUNC3
// This is an especially simple case for the phase-3 histogram, so it is
// implemented by itself here. It serves as a reference for the more intricate
// NUM_BITS < 6 case in hist3.cu.

__shared__ volatile uint bucketOffsets_shared3_64[MAX_BUCKETS];
__shared__ volatile uint columnScan_shared3_64[2 * NUM_THREADS];
__shared__ volatile uint gatherScan_shared3_64[2 * NUM_THREADS];
__shared__ volatile uint bucketCodes_shared3_64[5 * NUM_THREADS];

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void HISTOGRAM_FUNC3(const uint* bucketCount_global, 
	const uint2* rangePairs_global, const uint* countScan_global,
	const uint* columnScan_global, uint* bucketCodes_global,
	int supportEarlyExit) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	
	uint2 range = rangePairs_global[NUM_WARPS * block + warp];

	// The early exit flag comes in the most significant bit of the low short of
	// the first value in each sort block's radix digit frequency array.
	uint earlyExitBit = supportEarlyExit ? 1 : 0;

	// Find the starting scatter offsets for the first sort block in for each
	// warp in this histogram pass. Phase one has already computed the offsets
	// for the first sort block in each warp within this histogram block and 
	// cached the values in columnScan_global. We take those values and add the
	// global bucket scatter offsets from countScan_global.
	if(tid < NUM_BUCKETS)
		bucketOffsets_shared3_64[tid] = 
			countScan_global[NUM_BUCKETS * block + tid];
	
	uint globalIndex = 2 * NUM_THREADS * block + tid;

	columnScan_shared3_64[tid] = 
		columnScan_global[globalIndex];

	columnScan_shared3_64[NUM_THREADS + tid] = 
		columnScan_global[globalIndex + NUM_THREADS];

	__syncthreads();
	
	uint scatter1 = 
		columnScan_shared3_64[warp * NUM_CHANNELS + lane] +
		bucketOffsets_shared3_64[lane];
	uint scatter2 =
		columnScan_shared3_64[(NUM_WARPS + warp) * NUM_CHANNELS + lane] +
		bucketOffsets_shared3_64[NUM_CHANNELS + lane];
	__syncthreads();

	// Each thread loads a low and high counter.
	volatile uint* countWarp = columnScan_shared3_64 + 2 * WARP_SIZE * warp;
	volatile uint* count = countWarp + lane;

#ifdef INCLUDE_TRANSACTION_LIST
	volatile uint* gatherWarp = gatherScan_shared3_64 + 2 * WARP_SIZE * warp;
	volatile uint* gather = gatherWarp + lane;
#endif

	count[0] = scatter1;
	count[NUM_CHANNELS] = scatter2;

	uint current = range.x;
	while(current < range.y) {
		uint packed = bucketCount_global[WARP_SIZE * current + lane];

		uint sortedFlag = bfe(packed, 15, earlyExitBit);
		uint count1 = 0x00007fff & packed;
		uint count2 = packed>> 16;

		uint* bucketCodes = bucketCodes_global + TRANS_STRUCT_SIZE * current;

#ifdef INCLUDE_TRANSACTION_LIST

		bucketCodes[lane] = scatter1;// + sortedFlag;
		bucketCodes[WARP_SIZE + lane] = scatter2; 

		// Begin by computing the terms to put in the gather codes. It's
		// convenient to first find the next non-empty bucket to establish a 
		// transaction linked list. If the current bucket has no values, scan
		// right and find the first bucket that does.
		uint next1 = count1 ? lane : 0xffffffff;
		uint next2 = count2 ? (WARP_SIZE + lane) : 0xffffffff;
		count[0] = next1;
		count[WARP_SIZE] = next2;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			uint offset = 1<< i;
			uint y1 = count[offset];
			uint y2 = count[WARP_SIZE + offset];
			if(0xffffffff == next1) next1 = y1;
			if(0xffffffff == next2) next2 = y2;
			count[0] = next1;
			count[WARP_SIZE] = next2;
		}
		if(0xffffffff == next1) next1 = next2;
		count[0] = next1;

		// Grab the subsequent bucket
		next1 = count[1];
		next2 = count[WARP_SIZE + 1];		

		uint transCount1 = NumBucketTransactions(scatter1, count1);
		uint transCount2 = NumBucketTransactions(scatter2, count2);

		// Perform the parallel scan on both bucket counts and transaction 
		// counts by packing both terms into the same values.
		uint gather1 = bfi(count1, transCount1, 16, 16);
		uint gather2 = bfi(count2, transCount2, 16, 16);
#else
		uint gather1 = count1;
		uint gather2 = count2;
#endif
		count[0] = gather1;
		count[WARP_SIZE] = gather2;
		uint sum1 = gather1;
		uint sum2 = gather2;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			uint offset = 1<< i;
			uint y1 = count[-offset];
			uint y2 = count[WARP_SIZE - offset];
			if(lane >= offset) gather1 += y1;
			gather2 += y2;
			count[0] = gather1;
			count[WARP_SIZE] = gather2;
		}
		gather2 += gather1;
		count[WARP_SIZE] = gather2;

		gather1 -= sum1;
		gather2 -= sum2;

#ifndef INCLUDE_TRANSACTION_LIST	
		// Subtract gather index from scatter index. 
		bucketCodes[lane] = 4 * (scatter1 - gather1) + sortedFlag;
		bucketCodes[WARP_SIZE + lane] = 4 * (scatter2 - gather2);
#else 
		// INCLUDE_TRANSACTION_LIST

		uint transTotal = countWarp[NUM_BUCKETS - 1]>> 16;
		uint transOffset1 = gather1>> 16;
		uint transOffset2 = gather2>> 16;

		// Pack these three terms together to describe each bucket in the
		// transaction struct:
		// 14:0 - gather index into block shared memory.
		// 24:15 - number of transactions for data in this bucket.
		// 31:25 - index of the next non-zero bucket. establishes a linked list.
		if(!lane) gather1 = transTotal;
		uint bucketCode1 = (0xffff & gather1) |
			(transCount1<< 15) | (next1<< 25);
		uint bucketCode2 = (0xffff & gather2) | 
			(transCount2<< 15) | (next2<< 25);

		bucketCodes[2 * WARP_SIZE + lane] = bucketCode1;
		bucketCodes[3 * WARP_SIZE + lane] = bucketCode2;

		// Write the transaction offsets back into shared memory. We have to do
		// this to compute the first bucket transaction for this lane's list 
		// below.
		gather[0] = transOffset1;
		gather[WARP_SIZE] = transOffset2;

		// Map each bucket to a list. Note this requires integer division, a
		// very bad operation that inflates register count.
		uint list1 = MapBucketToList(transOffset1, transCount1, transTotal);
		uint list2 = MapBucketToList(transOffset2, transCount2, transTotal);

		// Scatter bucket index into list1/list2 if it's not 0xffffffff.
		count[0] = 0;
		count[WARP_SIZE] = 0;
		
		if(0xffffffff != list1) countWarp[list1] = lane;
		if(0xffffffff != list2) countWarp[list2] = WARP_SIZE + lane;

		// There are 32 transaction lists in the structure. If a bucket's
		// transactions start in a particular list, that bucket's index is
		// written to count at the list index. We need to scan from left to 
		// right to find which bucket to begin each list with. This simply 
		// requires us to find the last set bucket not to the right of the 
		// current list.
		uint firstBucket = count[0];
		#pragma unroll
		for(int i = 0; i < LOG_NUM_TRANSACTION_LISTS; ++i) {
			uint offset = 1<< i;
			uint y = count[-offset];
			uint y2 = max(firstBucket, y);
			if(offset <= lane) firstBucket = y2;
			count[0] = firstBucket;
		}

		// Compute the number of transactions in this lane's list.
		uint numListTrans = NumListTransactions(lane, transTotal);
		uint listTransOffset = ListTransactionOffset(lane, transTotal);
		uint firstBucketOffset = listTransOffset - gatherWarp[firstBucket];

		// Pack these three terms together to establish each list:
		// 6:0 - index of the bucket at which to begin global stores.
		// 16:7 - first transaction of the first bucket. After this, the
		//		warp traverses the buckets like a linked list
		// 31:17 - total number of transactions for this list to process.
		uint listCode = firstBucket | 
			(firstBucketOffset<< 7) | (numListTrans<< 17);
		bucketCodes[4 * WARP_SIZE + lane] = listCode;

#endif
		// update the scatter indices
		scatter1 += count1;
		scatter2 += count2;

		++current;
	}
}

