#include "params.cu"
#define NUM_WARPS NUM_COUNT_WARPS

#include "count.cu"

#define NUM_THREADS (NUM_WARPS * WARP_SIZE)
#define GATHER_SUM_MODE 1		// change to 2 for larger blocks.

#define INNER_LOOP 16

#define SINGLE_BLOCK_TASK false

////////////////////////////////////////////////////////////////////////////////
// Count kernels without early exit detection
/*
// 66% occupancy
GEN_COUNT_FUNC(CountBuckets_1, NUM_THREADS, 1, INNER_LOOP, 1, 8)
GEN_COUNT_FUNC(CountBuckets_2, NUM_THREADS, 2, INNER_LOOP, 1, 8)
GEN_COUNT_FUNC(CountBuckets_3, NUM_THREADS, 3, INNER_LOOP, 1, 8)
GEN_COUNT_FUNC(CountBuckets_4, NUM_THREADS, 4, INNER_LOOP, 1, 8)
GEN_COUNT_FUNC(CountBuckets_5, NUM_THREADS, 5, INNER_LOOP, 1, 8)

// 50% occupancy
GEN_COUNT_FUNC(CountBuckets_6, NUM_THREADS, 6, INNER_LOOP, 1, 6)

// 25% occupancy
GEN_COUNT_FUNC(CountBuckets_7, NUM_THREADS, 7, INNER_LOOP, 1, 3)

*/

// 66% occupancy
GEN_COUNT_LOOP(CountBucketsLoop_1, NUM_THREADS, 1, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK, 8)
GEN_COUNT_LOOP(CountBucketsLoop_2, NUM_THREADS, 2, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK,  8)
GEN_COUNT_LOOP(CountBucketsLoop_3, NUM_THREADS, 3, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK, 8)
GEN_COUNT_LOOP(CountBucketsLoop_4, NUM_THREADS, 4, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK,  8)
GEN_COUNT_LOOP(CountBucketsLoop_5, NUM_THREADS, 5, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK, 8)

// 50% occupancy
GEN_COUNT_LOOP(CountBucketsLoop_6, NUM_THREADS, 6, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK,  6)

// 25% occupancy
GEN_COUNT_LOOP(CountBucketsLoop_7, NUM_THREADS, 7, INNER_LOOP, 1,			\
	SINGLE_BLOCK_TASK,  3)



// TODO: use named barriers (or even regular syncthreads) to have a warp always
// be loading and a warp always be adding. They then switch tasks and continue.
// The loading thread requires no shared memory, so we get an easy increase in 
// occupancy.

/*
#define NUM_BITS 1
#define COUNT_FUNC CountBuckets_1
#define COUNT_SHARED_MEM count_shared1
#include "count.cu"

#define NUM_BITS 2
#define COUNT_FUNC CountBuckets_2
#define COUNT_SHARED_MEM count_shared2
#include "count.cu"

#define NUM_BITS 3
#define COUNT_FUNC CountBuckets_3
#define COUNT_SHARED_MEM count_shared3
#include "count.cu"

#define NUM_BITS 4
#define COUNT_FUNC CountBuckets_4
#define COUNT_SHARED_MEM count_shared4
#include "count.cu"

#define NUM_BITS 5
#define COUNT_FUNC CountBuckets_5
#define COUNT_SHARED_MEM count_shared5
#include "count.cu"

#define NUM_BITS 6
#define COUNT_FUNC CountBuckets_6
#define COUNT_SHARED_MEM count_shared6
#include "count.cu"


////////////////////////////////////////////////////////////////////////////////
// Count kernels with early exit detection

#define DETECT_SORTED

#define NUM_BITS 1
#define COUNT_FUNC CountBuckets_ee_1
#define COUNT_SHARED_MEM count_shared_ee1
#include "count.cu"

#define NUM_BITS 2
#define COUNT_FUNC CountBuckets_ee_2
#define COUNT_SHARED_MEM count_shared_ee2
#include "count.cu"

#define NUM_BITS 3
#define COUNT_FUNC CountBuckets_ee_3
#define COUNT_SHARED_MEM count_shared_ee3
#include "count.cu"

#define NUM_BITS 4
#define COUNT_FUNC CountBuckets_ee_4
#define COUNT_SHARED_MEM count_shared_ee4
#include "count.cu"

#define NUM_BITS 5
#define COUNT_FUNC CountBuckets_ee_5
#define COUNT_SHARED_MEM count_shared_ee5
#include "count.cu"

#define NUM_BITS 6
#define COUNT_FUNC CountBuckets_ee_6
#define COUNT_SHARED_MEM count_shared_ee6
#include "count.cu"
*/