#include "sortmultiscan.cu"


texture<uint4, cudaTextureType1D, cudaReadModeElementType> keys_texture_in;

template<int NumThreads, int NumBits, int ValuesPerThread, bool LoadFromTex>
DEVICE2 void ScatterFromKeys(const uint* keys_global_in, uint bit, 
	uint* scan_global_out) {

	const int NumValues = NumThreads * ValuesPerThread;
	const int NumDigits = 1<< NumBits;
	const int NumCounters = NumDigits / 2;
	
	const int TotalCounters = NumThreads * NumCounters;
	const int SegLen = TotalCounters / NumThreads + 1;
	const int NumScanValues = NumThreads * SegLen;
	const int NumRedValues = NumThreads / WARP_SIZE;
	const int TotalRedValues = WARP_SIZE * NumRedValues;
	const int ParallelScanSize = WARP_SIZE + WARP_SIZE / 2;

	const int ScratchSize = TotalCounters + NumScanValues + TotalRedValues +
		ParallelScanSize;
	__shared__ uint scratch_shared[ScratchSize];

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

	uint digits[ValuesPerThread];
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		digits[v] = bfe(keys[v], bit, NumBits);

	// Scan
	uint scatter[ValuesPerThread];
	MultiScanCounters<NumThreads, NumBits, ValuesPerThread>(
		tid, digits, scratch_shared, scatter);
/*
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		scan_global_out[NumValues * block + ValuesPerThread * tid + v] = 
			scatter[v];
	return;
*/
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		StoreShifted(scratch_shared, scatter[v], keys[v]);
	__syncthreads();

	// Store scan indices back to global mem.
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		scan_global_out[NumValues * block + ValuesPerThread * tid + v] = 
			scratch_shared[ValuesPerThread * tid + v];
}

#define GEN_MULTISCAN(Name, NumBits, ValuesPerThread, NumThreads)			\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, 4)						\
void Name(const uint* keys_global_in, uint bit,								\
	uint* scan_global_out) {												\
																			\
	ScatterFromKeys<NumThreads, NumBits, ValuesPerThread, true>(			\
		keys_global_in,	bit, scan_global_out);								\
}

// On nvcc 4.1: Uses 44 bytes of stackframe due to register spill with 16 vals.
// On --open64: COMPILES WITH NO SPILL! guys...

//GEN_MULTISCAN(MultiScan5_8_64, 5, 8, 64)
GEN_MULTISCAN(MultiScan5_16_64, 5, 16, 64)
//GEN_MULTISCAN(MultiScan5_8_128, 5, 8, 128)
//GEN_MULTISCAN(MultiScan5_16_128, 5, 16, 128)