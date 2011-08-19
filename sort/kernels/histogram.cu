#define NUM_BUCKETS (1<< NUM_BITS)
#define NUM_CHANNELS (NUM_BUCKETS / 2)

// The packed counters for this many sort blocks are packed in each warp of
// count values. NUM_CHANNELS * NUM_SORT_BLOCKS_PER_WARP gives the number of
// encoded channels in each warp.
#define NUM_SORT_BLOCKS_PER_WARP MIN(NUM_COUNT_WARPS, WARP_SIZE / NUM_CHANNELS)
#define NUM_HALF1_SORT_BLOCKS MIN(WARP_SIZE / NUM_BUCKETS, NUM_COUNT_WARPS)
#define NUM_HALF2_SORT_BLOCKS (NUM_SORT_BLOCKS_PER_WARP - NUM_HALF1_SORT_BLOCKS)

// Bucket counts come in packed - the first half of the bucket counts are in the
// low 16 bits and the second half are in the high 16 bits. Each thread 
// maintains two counters. For example, if NUM_BUCKETS = 32 and tid = 3, then 
// countLow tracks bucket 3 and countHigh tracks bucket 18.

// bucketCount_global is the array of interleaved bucket counts from count.cu

// histogram_global is a scratch array used in reduction. It is sized to
// NUM_BUCKETS * gridDim.x

// countScan_global is the output array of:

// Supports up to 16384 (NUM_THREADS=1024, VALUES_PER_THREAD=16) keys per block.
// The data structure up to NUM_BITS=7, although the histogram and count kernels
// only support up to NUM_BITS=6.

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

#include "hist1.cu"
#include "hist2.cu"

#ifdef INCLUDE_TRANSACTION_LIST
// For transaction lists, we need scatter and gather values for all buckets,
// plus a value for each transaction list.
#define TRANS_STRUCT_CONTENT (2 * NUM_BUCKETS + NUM_TRANSACTION_LISTS)
#define TRANS_STRUCT_SIZE ROUND_UP(TRANS_STRUCT_CONTENT, WARP_SIZE)
#else
// For simple scatter, we need only serialize a scatter offset for each bucket.
#define TRANS_STRUCT_CONTENT NUM_BUCKETS
#define TRANS_STRUCT_SIZE TRANS_STRUCT_CONTENT
#endif


#if 6 == NUM_BITS
#include "hist3_64.cu"
#else
#include "hist3.cu"
#endif

#undef NUM_BUCKETS
#undef NUM_BITS
#undef NUM_CHANNELS

#undef HISTOGRAM_FUNC1
#undef HISTOGRAM_FUNC2
#undef HISTOGRAM_FUNC3
#undef HISTOGRAM_BUILD
#undef TRANS_STRUCT_CONTENT
#undef TRANS_STRUCT_SIZE
#undef ComputeGatherStruct
#undef ComputeTransactionList
