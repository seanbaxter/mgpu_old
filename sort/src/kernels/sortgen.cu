// One of these macros will be defined:
// VALUE_TYPE_NONE
// VALUE_TYPE_INDEX
// VALUE_TYPE_SINGLE
// VALUE_TYPE_MULTI

// If VALUE_TYPE_NONE is defined, we are working directly on keys. Otherwise, we
// produce fused keys and keep the key values in a register array.

#ifdef VALUE_TYPE_NONE
	#define IS_SORT_KEY
#endif

#include "common.cu"

#define NUM_VALUES (VALUES_PER_THREAD * NUM_THREADS)

#ifdef BUILD_64

// 64bit kernels require more registers and so run less efficiently.
// To eliminate skills, give them so more regs. Note that this needs to be
// better optimized.

#ifdef VALUE_TYPE_NONE
#define REGS_PER_THREAD 32
#else
#define REGS_PER_THREAD 40
#endif


#else

#ifdef VALUE_TYPE_NONE
#define REGS_PER_THREAD 32
#else
#define REGS_PER_THREAD 36
#endif

#endif // BUILD_64


#define NUM_BLOCKS (32768 / ((~1 & (1 + REGS_PER_THREAD)) * NUM_THREADS))


// may keep any number of fused keys in shared memory while performing
// the scan - this reduces register pressure for key/value pairs.

#if 32 == NUM_WARPS
	#define LOG_NUM_WARPS 5
#elif 16 == NUM_WARPS
	#define LOG_NUM_WARPS 4
#elif 8 == NUM_WARPS
	#define LOG_NUM_WARPS 3
#elif 4 == NUM_WARPS
	#define LOG_NUM_WARPS 2
#endif


#include "sortcommon.cu"
#include "sortstore.cu"
#include "sortgeneric.cu"

#ifndef SCATTER_INPLACE

#define DETECT_SORTED

#define NUM_BITS 1
#define SORT_FUNC RadixSort_ee_1
#define SCATTER_LIST_NAME RadixSort_ee_1_scatter_shared
#include "sort.cu"

#define NUM_BITS 2
#define SORT_FUNC RadixSort_ee_2
#define SCATTER_LIST_NAME RadixSort_ee_2_scatter_shared
#include "sort.cu" 

#define NUM_BITS 3
#define SORT_FUNC RadixSort_ee_3
#define SCATTER_LIST_NAME RadixSort_ee_3_scatter_shared
#include "sort.cu"

#define NUM_BITS 4
#define SORT_FUNC RadixSort_ee_4
#define SCATTER_LIST_NAME RadixSort_ee_4_scatter_shared
#include "sort.cu"

#define NUM_BITS 5
#define SORT_FUNC RadixSort_ee_5
#define SCATTER_LIST_NAME RadixSort_ee_5_scatter_shared
#include "sort.cu"

#define NUM_BITS 6
#define SORT_FUNC RadixSort_ee_6
#define SCATTER_LIST_NAME RadixSort_ee_6_scatter_shared
#include "sort.cu"

#undef DETECT_SORTED

#endif // SCATTER_INPLACE

#define NUM_BITS 1
#define SORT_FUNC RadixSort_1
#define SCATTER_LIST_NAME RadixSort_1_scatter_shared
#include "sort.cu"

#define NUM_BITS 2
#define SORT_FUNC RadixSort_2
#define SCATTER_LIST_NAME RadixSort_2_scatter_shared
#include "sort.cu" 

#define NUM_BITS 3
#define SORT_FUNC RadixSort_3
#define SCATTER_LIST_NAME RadixSort_3_scatter_shared
#include "sort.cu"

#define NUM_BITS 4
#define SORT_FUNC RadixSort_4
#define SCATTER_LIST_NAME RadixSort_4_scatter_shared
#include "sort.cu"

#define NUM_BITS 5
#define SORT_FUNC RadixSort_5
#define SCATTER_LIST_NAME RadixSort_5_scatter_shared
#include "sort.cu"

#define NUM_BITS 6
#define SORT_FUNC RadixSort_6
#define SCATTER_LIST_NAME RadixSort_6_scatter_shared
#include "sort.cu"

