// One of these macros will be defined:
// VALUE_TYPE_NONE
// VALUE_TYPE_INDEX
// VALUE_TYPE_SINGLE
// VALUE_TYPE_MULTI

// If VALUE_TYPE_NONE is defined, we are working directly on keys. Otherwise, we
// produce fused keys and keep the key values in a register array.

#if defined(VALUE_TYPE_NONE)
	#define VALUE_COUNT 0
#elif defined(VALUE_TYPE_INDEX)
	#define VALUE_COUNT -1
#elif defined(VALUE_TYPE_SINGLE)
	#define VALUE_COUNT 1
#elif defined(VALUE_TYPE_MULTI)
	#define VALUE_COUNT 2
#endif

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


#define LOAD_FROM_TEXTURE true

texture<uint4, cudaTextureType1D, cudaReadModeElementType> keys_texture_in;

#include "sortloop.cu"

#define NUM_BLOCKS (32768 / ((~1 & (1 + REGS_PER_THREAD)) * NUM_THREADS))

GEN_SORT_LOOP(RadixSortLoop_1, NUM_THREADS, 1, VALUE_COUNT,					\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_2, NUM_THREADS, 2, VALUE_COUNT,					\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_3, NUM_THREADS, 3, VALUE_COUNT,					\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_4, NUM_THREADS, 4, VALUE_COUNT,					\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_5, NUM_THREADS, 5, VALUE_COUNT, 				\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_6, NUM_THREADS, 6, VALUE_COUNT,					\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)
GEN_SORT_LOOP(RadixSortLoop_7, NUM_THREADS, 7, VALUE_COUNT, 				\
	LOAD_FROM_TEXTURE, NUM_BLOCKS)

