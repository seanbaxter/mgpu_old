#pragma once

#include "common.cu"

// counterSize is 8 (for byte-packed) or 16 (for short-packed).
// strided is false to put digits in order (0, 1), (2, 3), (4, 5), etc.
// strided is true to put digits in order (0, 16), (1, 17), (2, 18), etc.

DEVICE volatile uint* PackedCounterRef(uint digit, uint numDigits, 
	uint numThreads, volatile uint* counters_shared, int counterSize, 
	bool strided, uint& shift) {

	volatile uint* counter;
	if(8 == counterSize) {
		if(strided) {
			uint index = numThreads * ((numDigits / 4 - 1) & digit);
			shift = 8 * (digit / 4);
			counter = &counters_shared[index];
		} else {
			uint index = numThreads * (digit / 4);
			shift = 8 * (3 & digit);
			counter = &counters_shared[index];
		}
	} else if(16 == counterSize) { 
		if(strided) {
			uint index = numThreads * ((numDigits / 2 - 1) & digit);
			shift = 16 * (digit / 2);
			counter = &counters_shared[index];
		} else {
			uint index = numThreads * (digit / 2);
			shift = 16 * (1 & digit);
			counter = &counters_shared[index];
		}	
	}
	return counter;
}

// Returns the counter content before the increment.
DEVICE uint IncPackedCounter(uint digit, uint numDigits, uint numThreads,
	volatile uint* counters_shared, int counterSize, bool strided,
	uint value) {

	uint shift;
	volatile uint* p = PackedCounterRef(digit, numDigits, numThreads,
		counters_shared, counterSize, strided, shift);
	uint counter = *p;
	*p = counter + (value<< shift);

	return counter;
}

DEVICE uint GatherPackedCounter(uint digit, uint numDigits, uint numThreads,
	volatile uint* counters_shared, int counterSize, bool strided,
	uint& shift) {

	volatile uint* p = PackedCounterRef(digit, numDigits, numThreads,
		counters_shared, counterSize, strided, shift);
	return *p;
}

template<int NumThreads, int NumBits, int ValuesPerThread, int NumRakingThreads>
DEVICE2 void MultiScanCounters(uint tid, const uint* digits, 
	volatile uint* scratch_shared, uint* scatter) {

	const int NumDigits = 1<< NumBits;

	// Pack two counters per uint (16-bit packing).
	const int NumCounters = NumDigits / 2;
	const int TotalCounters = NumThreads * NumCounters;

	// Evenly divide the TotalCounters counters over all raking threads..
	// However add 1 to make SegLen relatively prime to the number of banks to
	// avoid bank conflicts. For NumDigits = 32, NumCounters = 16, and 
	// NumThreads = NumRakingThreads = 128, each thread sequentially scans 16
	// counters, but this number is bumped up by 1:

	// eg,
	// tid 0 processes bank 0 on cycle 0
	// tid 1 processes bank 17 on cycle 0
	// tid 2 processes bank 2 on cycle 0
	// tid 3 process bank 19 on cycle 0, etc.

	// Parallel scan is handled from a single warp. 128 reduction totals are 
	// stored to shared memory, and strided. Each thread of warp 0 loads four
	// of these values, sums them up, and runs a simple 5-level parallel scan.
	// The four reduction values are then subtracted out in reverse order, and
	// each difference is stored back to the shared mem location from which its
	// reduction value was loaded.

	// For a block with 128 threads and 16 values per thread, these 2048 values
	// generate 2048 digit counters (for NumBits = 5), but they are scanned with
	// sequential scans and just a single simple intra-warp parallel scan.

	const int SegLen = TotalCounters / NumRakingThreads + 1;
	const int NumScanValues = NumRakingThreads * SegLen;

	const int NumRedValues = NumRakingThreads / WARP_SIZE;
	const int TotalRedValues = WARP_SIZE * NumRedValues;

	uint warp = tid / WARP_SIZE;

	volatile uint* counters_shared = scratch_shared;
	volatile uint* reduction_shared = scratch_shared + NumScanValues;
	volatile uint* scan_shared = reduction_shared + 
		(TotalRedValues + TotalRedValues / WARP_SIZE);

	// Clear the counters.
	#pragma unroll
	for(int i = 0; i < NumCounters; ++i)
		counters_shared[i * NumThreads] = 0;

	// Clear the padding counters at the end.
	scratch_shared[SegLen * NumRakingThreads - NumThreads + tid] = 0;

	// Compute the digit counts and save the thread-local scan per digit.
	// NOTE: may need to bfi into localScan to fight register pressure.
	// Shift and add in a 4 for each digit occurence to eliminate some pointer
	// arithmetic due to Fermi not having a mov/lea-like 4 * mul in STS/LDS.
	uint localScan[ValuesPerThread];
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		localScan[v] = IncPackedCounter(digits[v], NumDigits, NumThreads, 
			counters_shared + tid, 16, true, 4);
	__syncthreads();

	uint scanValues[SegLen];
	if(tid < NumRakingThreads) {

		uint x = 0;
		volatile uint* seg_shared = counters_shared + SegLen * tid;
		#pragma unroll
		for(int i = 0; i < SegLen; ++i) {
			scanValues[i] = x;
			x += seg_shared[i];
		}

		// Store the counters with stride.
		reduction_shared[tid + warp] = x;
	}
	__syncthreads();

	// Scan from a single warp.
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
		volatile uint* s = scan_shared + tid;
		scan_shared[-(WARP_SIZE / 2)] = 0;

		uint x = sum;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			x += s[-offset];
			s[0] = x;
		}

		// Add in the reduction of the top row to all elements in the bottom
		// row of the packed scan.
		x += s[WARP_SIZE + WARP_SIZE / 2 - 1]<< 16;

		// Subtract out the threadVals to get an exclusive scan and store.
		#pragma unroll
		for(int i = NumRedValues - 1; i >= 0; --i) {
			x -= threadVals[i];
			reduction_shared[index + i] = x;
		}
	}
	__syncthreads();

	// Add the scanned values back into the stored scanValues.
	if(tid < NumRakingThreads) {
		uint offset = reduction_shared[tid + warp];
		volatile uint* seg_shared = counters_shared + SegLen * tid;

		#pragma unroll
		for(int i = 0; i < SegLen; ++i)
			seg_shared[i] = scanValues[i] + offset;
	}
	__syncthreads();

	// Gather the scanned offsets for each digit and add in the local offsets
	// saved in localScan.
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v) {
		uint shift;
		uint digitScan = GatherPackedCounter(digits[v], NumDigits, NumThreads,
			counters_shared + tid, 16, true, shift);

		scatter[v] = bfe(localScan[v] + digitScan, shift, 16);
	}
	__syncthreads();
}



template<int NumThreads, int NumBits, int ValuesPerThread>
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
	uint digits[ValuesPerThread];
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v) {
		keys[v] = keys_global_in[NumValues * block + ValuesPerThread * tid + v];
		digits[v] = bfe(keys[v], bit, NumBits);
	}

	// Scan
	uint scatter[ValuesPerThread];
	MultiScanCounters<NumThreads, NumBits, ValuesPerThread, NumThreads>(
		tid, digits, scratch_shared, scatter);

	// Store scan indices back to global mem.
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		scan_global_out[NumValues * block + ValuesPerThread * tid + v] = 
			scatter[v];
}

#define GEN_MULTISCAN(Name, NumBits, ValuesPerThread, NumThreads)			\
																			\
extern "C" __global__ void Name(const uint* keys_global_in, uint bit,		\
	uint* scan_global_out) {												\
																			\
	ScatterFromKeys<NumThreads, NumBits, ValuesPerThread>(keys_global_in,	\
		bit, scan_global_out);												\
}

// Spills with 16 values - compiles with 12?!
GEN_MULTISCAN(MultiScan5_16_128, 5, 12, 128)

