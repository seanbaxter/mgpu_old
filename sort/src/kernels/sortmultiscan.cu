#pragma once

#include "common.cu"

// counterSize is 8 (for byte-packed) or 16 (for short-packed).
// strided is false to put digits in order (0, 1), (2, 3), (4, 5), etc.
// strided is true to put digits in order (0, 16), (1, 17), (2, 18), etc.

template<int X> struct LogPow2Const { 
	static const int value = LogPow2Const<X / 2>::value + 1;
};
template<> struct LogPow2Const<1> { 
	static const int value = 0;
};

template<int NumDigits, int NumThreads>
DEVICE2 volatile uint* PackedCounterRef(uint digit, uint tid, 
	volatile uint* counters_shared, int counterSize, bool strided, 
	uint& shift) {

	const int NumBits = LogPow2Const<NumDigits>::value;
	const int LogThreads = LogPow2Const<NumThreads>::value;

	volatile uint* counter;
	if(8 == counterSize) {
		if(strided) {
			uint index = NumThreads * ((NumDigits / 4 - 1) & digit);
			shift = 8 * (digit / (NumDigits / 4));
			counter = &counters_shared[index + tid];
		} else {
			uint index = NumThreads * (digit / 4);
			shift = 8 * (3 & digit);
			counter = &counters_shared[index + tid];
		}
	} else if(16 == counterSize) { 
		if(strided) {
			// If NumDigits is 32 we can just mask out the least sig four bits
			// to accelerate 16 * (digit / (NumDigits / 2)).
			if(32 == NumDigits) shift = 16 & digit;
			else shift = 16 * (digit / (NumDigits / 2));
		
			// uint index = 16 * (digit / (NumDigits / 2));
			// We can accelerate that by using bfi to insert the least sig bits
			// of the digit into the pre-scaled tid. Note that we need to pass
			// counters_shared as a CONSTANT pointer from the start of the shmem
			// array for the block, as Fermi doesn't allow you to add two 
			// registers together in an LDS or STS statement.
			uint index = bfi(4 * tid, digit, 2 + LogThreads, NumBits - 1);
			counter = (volatile uint*)((volatile char*)counters_shared + index);
		} else {
			uint index = NumThreads * (digit / 2);
			shift = 16 * (1 & digit);
			counter = &counters_shared[index + tid];
		}
	}
	return counter;
}

// Returns the counter content before the increment.
template<int NumDigits, int NumThreads>
DEVICE2 uint IncPackedCounter(uint digit, uint tid,
	volatile uint* counters_shared, int counterSize, bool strided, 
	uint value) {

	uint shift;
	volatile uint* p = PackedCounterRef<NumDigits, NumThreads>(digit, tid,
		counters_shared, counterSize, strided, shift);
	uint counter = *p;
	*p = counter + (value<< shift);

	return counter;
}

template<int NumDigits, int NumThreads>
DEVICE2 uint GatherPackedCounter(uint digit, uint tid,
	volatile uint* counters_shared, int counterSize, bool strided, 
	uint& shift) {

	volatile uint* p = PackedCounterRef<NumDigits, NumThreads>(digit, tid,
		counters_shared, counterSize, strided, shift);
	return *p;
}

template<int NumThreads, int NumBits, int ValuesPerThread>
DEVICE2 void MultiScanCounters(uint tid, const uint* digits, 
	volatile uint* scratch_shared, uint* scatter) {

	const int NumRakingThreads = NumThreads;
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
		counters_shared[i * NumThreads + tid] = 0;

	// Clear the padding counters at the end.
	scratch_shared[SegLen * NumRakingThreads - NumThreads + tid] = 0;

	// Compute the digit counts and save the thread-local scan per digit.
	// NOTE: may need to bfi into localScan to fight register pressure.
	// Shift and add in a 4 for each digit occurence to eliminate some pointer
	// arithmetic due to Fermi not having a mov/lea-like 4 * mul in STS/LDS.
	uint localScan[ValuesPerThread];
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v)
		localScan[v] = IncPackedCounter<NumDigits, NumThreads>(digits[v], tid,
			counters_shared, 16, true, 4);
	__syncthreads();

	// Add up all the packed counters in this segment. We would prefer to load
	// them once and create an exclusive sequential scan in register array, but
	// there aren't enough registers to allow this. Instead, we load them a 
	// second time after the parallel scan and do the running sum.
	uint x = 0;
	volatile uint* seg_shared = scratch_shared + SegLen * tid;
	#pragma unroll
	for(int i = 0; i < SegLen; ++i)
		x += seg_shared[i];

	// Store the counters with stride.
	reduction_shared[tid + warp] = x;

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

		// Subtract out the threadVals to get an exclusive scan and store.
		#pragma unroll
		for(int i = NumRedValues - 1; i >= 0; --i) {
			x -= threadVals[i];
			reduction_shared[index + i] = x;
		//	x -= reduction_shared[index + i];
		//	reduction_shared[index + i] = x;
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

	// Gather the scanned offsets for each digit and add in the local offsets
	// saved in localScan.
	#pragma unroll
	for(int v = 0; v < ValuesPerThread; ++v) {
		uint shift;
		uint digitScan = GatherPackedCounter<NumDigits, NumThreads>(digits[v],
			tid, counters_shared, 16, true, shift);

		scatter[v] = bfe(localScan[v] + digitScan, shift, 16);
	}
	__syncthreads();
}



