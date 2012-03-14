#pragma once

#include "common.cu"

template<int NumThreads, int NumBits>
DEVICE2 volatile uint* PackedCounterRef(uint digit, uint tid, 
	volatile uint* counters_shared, uint& shift) {

	const int LogThreads = LogPow2Const<NumThreads>::value;
	const int NumDigits = 1<< NumBits;

	if(5 == NumBits) shift = 16 & digit;
	else shift = 16 * (digit / (NumDigits / 2));

	uint index = bfi(4 * tid, digit, 2 + LogThreads, NumBits - 1);
	return (volatile uint*)((volatile char*)counters_shared + index);
}

template<int NumThreads, int NumBits, int VT>
DEVICE2 void MultiScanCounters(uint tid, const uint* keys, uint bit, 
	volatile uint* scratch_shared, uint* scatter) {

	const int NumCounters = (1<< NumBits) / 2;
	const int SegLen = NumCounters + 1;
	const int NumScanValues = NumThreads * SegLen;
	const int RedFootprint = NumThreads + NumThreads / WARP_SIZE;
	const int NumRedValues = NumThreads / WARP_SIZE;	// num warps

	uint warp = tid / WARP_SIZE;

	volatile uint* counters_shared = scratch_shared;
	volatile uint* reduction_shared = scratch_shared + NumScanValues;
	volatile uint* scan_shared = reduction_shared + RedFootprint;

	// Clear the counters.
	#pragma unroll
	for(int i = 0; i < NumCounters; ++i)
		counters_shared[i * NumThreads + tid] = 0;

	// Clear the padding counters at the end.
	scratch_shared[SegLen * NumThreads - NumThreads + tid] = 0;

	uint digits[VT];
	uint localScan[VT];
	#pragma unroll
	for(int v = 0; v < VT; ++v) {
		digits[v] = bfe(keys[v], bit, NumBits);
		uint shift;
		volatile uint* p = PackedCounterRef<NumThreads, NumBits>(digits[v], tid,
			counters_shared, shift);
		localScan[v] = *p;
		*p = shl_add(1, shift, localScan[v]);
	}
	__syncthreads();

	// Add up all the packed counters in this segment. We would prefer to load
	// them once and create an exclusive sequential scan in register array, but
	// there aren't enough registers to allow this. Instead, we load them a 
	// second time after the parallel scan and do the running sum.
	volatile uint* seg_shared = scratch_shared + SegLen * tid;
	uint x = 0;
	#pragma unroll
	for(int i = 0; i < SegLen; ++i)
		x += seg_shared[i];

	// Store the counters with stride.
	reduction_shared[tid + warp] = x;

	__syncthreads();

	// Parallel scan from a single warp.
	if(tid < WARP_SIZE) {
		uint index = NumRedValues * tid;
		index += index / WARP_SIZE;
		
		uint threadVals[NumRedValues];
		uint sum = 0;
		#pragma unroll
		for(int i = 0; i < NumRedValues; ++i) {
			threadVals[i] = reduction_shared[index + i];
			sum += threadVals[i];
		}

		// Run a single parallel scan.
		volatile uint* s = scan_shared + tid + WARP_SIZE / 2;
		s[-(WARP_SIZE / 2)] = 0;
		s[0] = sum;

		uint x = sum;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			x += s[-offset];
			s[0] = x;
		}

		// Add in the reduction of the top row to all elements in the bottom
		// row of the packed scan.
		x += scan_shared[WARP_SIZE + WARP_SIZE / 2 - 1]<< 16;
		#pragma unroll
		for(int i = NumRedValues - 1; i >= 0; --i) {
			x -= threadVals[i];
			reduction_shared[index + i] = x;
		}
	}
	__syncthreads();

	// Add the scanned values back into the stored scanValues.
	x = 0;
	uint offset = reduction_shared[tid + warp];
	#pragma unroll
	for(int i = 0; i < SegLen; ++i)  {
		uint scanValue = seg_shared[i];
		seg_shared[i] = x + offset;
		x += scanValue;
	}
	__syncthreads();

	#pragma unroll
	for(int v = 0; v < VT; ++v) {
		uint shift;
		uint offset = *PackedCounterRef<NumThreads, NumBits>(
			bfe(keys[v], bit, NumBits), tid, counters_shared, shift);
		scatter[v] = bfe(localScan[v] + offset, shift, 16);
	}
	__syncthreads();
}

texture<uint4, cudaTextureType1D, cudaReadModeElementType> keys_texture_in;

template<int NumThreads, int NumBits, int VT>
DEVICE2 void SortFunc(const uint* keys_global_in, 
	const uint* bucketCodes_global, uint bit, uint* keys_global_out) {

	const int NumValues = NumThreads * VT;
	const int NumWarps = NumThreads / WARP_SIZE;
	const int SharedSize = 24 * NumThreads;	// target 33% occupancy.
	
	__shared__ uint shared[SharedSize];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	uint keys[VT];

	#pragma unroll
	for(int i = 0; i < VT / 4; ++i) {
		uint4 k = tex1Dfetch(keys_texture_in, 
			(NumValues * block + VT * tid) / 4 + i);
		keys[4 * i + 0] = k.x;
		keys[4 * i + 1] = k.y;
		keys[4 * i + 2] = k.z;
		keys[4 * i + 3] = k.w;	
	}

	// Run a local sort and return the keys into register in block-strided 
	// order.
	uint scatter[VT];
	MultiScanCounters<NumThreads, NumBits, VT>(tid, keys, bit, shared, scatter);

	// Scatter to shared memory with stride.
	#pragma unroll
	for(int v = 0; v < VT; ++v) {
		uint index = shr_add(scatter[v], LOG_WARP_SIZE, scatter[v]);
		shared[index] = keys[v];
	}
	__syncthreads();

	// Read back in thread order with stride and store to global memory.
	#pragma unroll
	for(int v = 0; v < VT; ++v) {
		uint index = NumThreads * v + tid;
		keys_global_out[NumValues * block + index] = 
			shared[(NumThreads + NumWarps) * v + tid + tid / WARP_SIZE];
	}
}

extern "C" __global__ void SortFunc5_128_16(const uint* keys_global_in, 
	const uint* bucketCodes_global, uint bit, uint* keys_global_out) {

	SortFunc<128, 5, 16>(keys_global_in, bucketCodes_global, bit, 
		keys_global_out);
}
