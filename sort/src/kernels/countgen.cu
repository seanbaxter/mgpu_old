#include "params.cu"
#define NUM_WARPS NUM_COUNT_WARPS

#include "countcommon.cu"

#define NUM_THREADS (NUM_WARPS * WARP_SIZE)
#define GATHER_SUM_MODE 1		// change to 2 for larger blocks.


__device__ uint sortDetectCounters_global[4];


////////////////////////////////////////////////////////////////////////////////
// Count kernels without early exit detection

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
