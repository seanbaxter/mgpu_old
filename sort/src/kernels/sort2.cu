#include "sortlocal.cu"


texture<uint4, cudaTextureType1D, cudaReadModeElementType> keys_texture_in;

template<int NumThreads, int NumBits, int ValuesPerThread, bool LoadFromTex,
	bool RecalcDigits, bool StoreInPlace, bool ExtraScatterArray>
DEVICE2 void SortFunc(const uint* keys_global_in, 
	const uint* bucketCodes_global, uint bit, uint* keys_global_out,
	uint numValuesStreams, uint* debug_global_out, 
	const uint* values1_global_in, const uint* values2_global_in,
	const uint* values3_global_in, const uint* values4_global_in,
	const uint* values5_global_in, const uint* values6_global_in,
	// For VALUE_TYPE_MULTI, we have to pass each of the pointers in as 
	// individual arguments, not as arrays, or CUDA generates much worse code,
	// full of unified addressing instructions.
	uint* values1_global_out, uint* values2_global_out,
	uint* values3_global_out, uint* values4_global_out,
	uint* values5_global_out, uint* values6_global_out) {

	const int NumValues = NumThreads * ValuesPerThread;
	const int SharedSize = 
		LocalSortSize<NumThreads, ValuesPerThread, NumBits>::SharedSize;
	const int NumDigits = 1<< NumBits;

	__shared__ uint scratch_shared[SharedSize];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	// Load the keys into register. Lots of bank conflicts but this is test
	// function so no matter.
	uint keys[ValuesPerThread];

	if(LoadFromTex) {
		const uint keysOffset = NumValues * block + ValuesPerThread * tid;
		#pragma unroll
		for(int i = 0; i < ValuesPerThread / 4; ++i) {
			uint4 k = tex1Dfetch(keys_texture_in, keysOffset / 4 + i);
			keys[4 * i + 0] = k.x;
			keys[4 * i + 1] = k.y;
			keys[4 * i + 2] = k.z;
			keys[4 * i + 3] = k.w;	
		}
	} else {
		const uint keysOffset = NumValues * block + ValuesPerThread * tid;
		#pragma unroll
		for(int i = 0; i < ValuesPerThread; ++i)
			keys[i] = keys_global_in[keysOffset + i];
	}

	// Load the scaled scatter offsets.
	__shared__ uint globalScatter_shared[NumDigits];
	uint globalScatter1, globalScatter2;
	if(!StoreInPlace) {
		if(ExtraScatterArray) {
			if(tid < NumDigits)
				globalScatter_shared[tid] = 
					bucketCodes_global[NumDigits * block + tid];
			if(NumThreads < NumDigits)
				globalScatter_shared[NumThreads + tid] = 
					bucketCodes_global[NumDigits * block + NumThreads + tid];
		} else {
			if(tid < NumDigits)
				globalScatter1 = bucketCodes_global[NumDigits * block + tid];
			if(NumThreads < NumDigits)
				globalScatter2 =
					bucketCodes_global[NumDigits * block + NumThreads + tid];
		}			
	}	

	// Run a local sort and return the keys into register in block-strided 
	// order.
	SortLocal<NumThreads, NumBits, ValuesPerThread>(tid, keys, bit, 
		scratch_shared, RecalcDigits);

	if(StoreInPlace) {
		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v) {
			uint index = NumThreads * v + tid;
			keys_global_out[NumValues * block + index] = keys[v];
		}
	} else {
		if(!ExtraScatterArray) {
			__syncthreads();
			scratch_shared[tid] = globalScatter1;
			if(NumThreads < NumDigits)
				scratch_shared[NumThreads + tid] = globalScatter2;
			__syncthreads();
		}

		// Load and scatter to global memory.
		#pragma unroll
		for(int v = 0; v < ValuesPerThread; ++v) {
			uint digit = bfe(keys[v], bit, NumBits);

			uint index = 4 * (NumThreads * v + tid);
			uint scatter = ExtraScatterArray ? globalScatter_shared[digit] : 
				scratch_shared[digit];

			StoreToGlobal2(keys_global_out, scatter, index, keys[v]);
		}
	}
}

#define GEN_RADIX_SORT(Name, NumBits, ValuesPerThread, NumThreads,			\
	BlocksPerSM, RecalcDigits, StoreInPlace, ExtraScatterArray)				\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global_in, const uint* bucketCodes_global,		\
	uint bit, uint* keys_global_out) {										\
																			\
	SortFunc<NumThreads, NumBits, ValuesPerThread, true,					\
		RecalcDigits, StoreInPlace, ExtraScatterArray>(						\
		keys_global_in,	bucketCodes_global, bit, keys_global_out,			\
			0, 0,															\
			0, 0, 0, 0, 0, 0,												\
			0, 0, 0, 0, 0, 0);												\
}


// On nvcc 4.1: Uses 44 bytes of stackframe due to register spill with 16 vals.
// On --open64: COMPILES WITH NO SPILL! guys...

//GEN_MULTISCAN(MultiScan5_8_64, 5, 8, 64)
  
#define STORE_IN_PLACE false

#define BLOCKS_PER_SM (512 / NUM_THREADS)

#define RECALC_DIGITS (VALUES_PER_THREAD > 16)

#define EXTRA_SCATTER (16 == VALUES_PER_THREAD)

GEN_RADIX_SORT(RadixSort_1, 1, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER)
GEN_RADIX_SORT(RadixSort_2, 2, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER)
GEN_RADIX_SORT(RadixSort_3, 3, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER)
GEN_RADIX_SORT(RadixSort_4, 4, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER)
GEN_RADIX_SORT(RadixSort_5, 5, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER)
GEN_RADIX_SORT(RadixSort_6, 6, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER || (128 == NUM_THREADS))
GEN_RADIX_SORT(RadixSort_7, 7, VALUES_PER_THREAD, NUM_THREADS, BLOCKS_PER_SM, \
	RECALC_DIGITS, STORE_IN_PLACE, EXTRA_SCATTER || (128 == NUM_THREADS))

