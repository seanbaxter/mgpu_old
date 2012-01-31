
#include "common.cu"

__device__ uint sortDetectCounters_global[4];


////////////////////////////////////////////////////////////////////////////////
// IncBucketCounter

DEVICE void IncBucketCounter(uint bucket, volatile uint* counters, 
	uint& counter0, uint& counter1, uint numBits) {

	if(1 == numBits)
		// For 1-bit keys, use 16-bit counters in a single register.
		// counter0 += 1<< (16 * bucket);
		counter0 = shl_add(1, 16 * bucket, counter0);
	else if(2 == numBits)
		// For 2-bit keys, use 8-bit counters in a single register.
		counter0 = shl_add(1, 8 * bucket, counter0);
	else if(3 == numBits) {
		// For 3-bit keys, use 8-bit counters in two registers.
		
		// Insert the least-significant 2 bits of bucket into bits 4:3 of shift.
		// That is, shift = 0, 8, 16, or 24 when bits 1:0 of bucket are 0, 1, 2,
		// or 3. bfi is available on Fermi for performing a mask and shift in
		// one instruction.
		uint shift = bfi(0, bucket, 3, 2);
		uint bit = 1<< shift;

		// Increment counter0 or counter1 by 1<< shift depending on bit 2 of
		// bucket.
		if(0 == (4 & bucket)) counter0 += bit;
		else counter1 += bit;

	} else {
		// For 4-, 5-, and 6-bit keys, use 8-bit counters in indexable shared
		// memory. This requires a read-modify-write to update the previous
		// count.

		// We can find the array index by dividing bucket by 4 (the number of
		// counters packed into each uint) and multiplying by 32 (the stride
		// between consecutive counters for the thread).
		// uint index = 32 * (bucket / 4);
		// It's likely more efficient to use a mask rather than a shift (the
		// NVIDIA docs claim shift is 2-cycles, but it is likely just one for
		// constant shift, so use a mask to be safe).

		uint index = (32 / 4) * (~3 & bucket);
		uint counter = counters[index];
		counter = shl_add(1, bfi(0, bucket, 3, 2), counter);
		counters[index] = counter;
	}
}


////////////////////////////////////////////////////////////////////////////////
// GatherSums

// Consider data in this format:
// 0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4  0 4 0 4 0 4 0 4 0 4 0 4 0 4 0 4 
// 1 5 1 5 1 5 1 5 1 5 1 5 1 5 1 5  1 5 1 5 1 5 1 5 1 5 1 5 1 5 1 5
// 2 6 2 6 2 6 2 6 2 6 2 6 2 6 2 6  2 6 2 6 2 6 2 6 2 6 2 6 2 6 2 6 
// 3 7 3 7 3 7 3 7 3 7 3 7 3 7 3 7  3 7 3 7 3 7 3 7 3 7 3 7 3 7 3 7

// GatherSums adds the bottom halves and top halves and re-orders:
// 0 2 4 6 0 2 4 6 0 2 4 6 0 2 4 6  0 2 4 6 0 2 4 6 0 2 4 6 0 2 4 6 
// 1 3 5 7 1 3 5 7 1 3 5 7 1 3 5 7  1 3 5 7 1 3 5 7 1 3 5 7 1 3 5 7

// The number of rows has been cut in half, so the following call will add and
// re-order for this:
// 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7  0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7

// A last parallel scan pass gets the final counter values, as there is no more
// sequential scan opportunity to exploit.

// The warp is cut into halves - threads in the first half add the top rows of 
// the right half columns into their own top halevs, and write at 2 * lane.
// Threads in the right half add the bottom rows of the left half into their own
// bottom halves, and write at 2 * lane + 1 - WARP_SIZE.

// mode behavior:
// 0 - simply add the values together and store in the target column
// 1 - add the values together, then unpack into ushorts and store the
//		low values at i and the high values at i + halfHeight
// 2 - unpack the values into ushorts, then add and store like in 1.

// If valuesPerThread >= 128, uncomment this. This define is safe when
// NUM_VALUES <= 2048.

// Have to use a template here because of bugs in the compiler. We could 
// allocate an array of registers on the count kernel side and pass them by
// "address" to GatherSums, but then NVCC generates dynamic local stores, 
// instead of static shared stores. With the template parameter we can allocate
// temporary register storage inside GatherSums.

template<int ColHeight>
DEVICE2 void GatherSums(uint lane, int mode, volatile uint* data) {

	uint targetTemp[MAX(ColHeight, 1)];

	uint sourceLane = lane / 2;
	uint halfHeight = ColHeight / 2;
	uint odd = 1 & lane;

	// Swap the two column pointers to resolve bank conflicts. Even columns read
	// from the left source first, and odd columns read from the right source 
	// first. All these support terms need only be computed once per lane. The 
	// compiler should eliminate all the redundant expressions.
	volatile uint* source1 = data + sourceLane;
	volatile uint* source2 = source1 + WARP_SIZE / 2;
	volatile uint* sourceA = odd ? source2 : source1;
	volatile uint* sourceB = odd ? source1 : source2;

	// Adjust for row. This construction should let the compiler calculate
	// sourceA and sourceB just once, then add odd * colHeight for each 
	// GatherSums call.
	uint sourceOffset = odd * (WARP_SIZE * halfHeight);
	sourceA += sourceOffset;
	sourceB += sourceOffset;
	volatile uint* dest = data + lane;
	
	#pragma unroll
	for(int i = 0; i < halfHeight; ++i) {
		uint a = sourceA[i * WARP_SIZE];
		uint b = sourceB[i * WARP_SIZE];

		if(0 == mode)
			targetTemp[i] = a + b;
		else if(1 == mode) {
			uint x = a + b;
			uint x1 = prmt(x, 0, 0x4140);
			uint x2 = prmt(x, 0, 0x4342);
			targetTemp[2 * i] = x1;
			targetTemp[2 * i + 1] = x2;
		} else if(2 == mode) {
			uint a1 = prmt(a, 0, 0x4140);
			uint a2 = prmt(a, 0, 0x4342);
			uint b1 = prmt(b, 0, 0x4140);
			uint b2 = prmt(b, 0, 0x4342);
			uint x1 = a1 + b1;
			uint x2 = a2 + b2;
			targetTemp[2 * i] = x1;
			targetTemp[2 * i + 1] = x2;
		}
	}

	#pragma unroll
	for(int i = 0; i < ColHeight / 2; ++i)
		dest[i * WARP_SIZE] = targetTemp[i];

	if(mode > 0) {
		#pragma unroll
		for(int i = 0; i < ColHeight / 2; ++i)
			dest[(i + halfHeight) * WARP_SIZE] = targetTemp[i + halfHeight];
	}
}


////////////////////////////////////////////////////////////////////////////////
// GatherSumsReduce

// For values/block <= 2048, Mode = 1 allows us to add then unpack values.
// For 4096 values/block, set Mode = 2.
template<int NumBits, int Mode>
DEVICE2 void GatherSumsReduce(volatile uint* warpCounters_shared, uint lane,
	volatile uint* counts_shared) {

	const int NumBuckets = 1<< NumBits;
	const int NumCounters = DIV_UP(NumBuckets, 4);
	const int NumChannels = DIV_UP(NumBuckets, 2);

	volatile uint* counters_shared = warpCounters_shared + lane;

	// If there are multiple counters, run GatherSums until we only have one
	// row of counters remaining.
	if(1 == NumBits) {
		// 16-bit packing was used from the start, so we go directly to parallel 
		// reduction.
	} else if(2 == NumBits) {
		// Grab even and odd counters, and add and widen
		uint a = warpCounters_shared[~1 & lane];
		uint b = warpCounters_shared[1 | lane];
		uint gather = (1 & lane) ? 0x4342 : 0x4140;
		uint sum = prmt(a, 0, gather) + prmt(b, 0, gather);
		counters_shared[0] = sum;
	} else if(3 == NumBits) {
		// At this point we have to manually unroll the GatherSums calls, 
		// because nvcc is stupid and complains "Advisory: Loop was not 
		//  unrolled, not an innermost loop." This is due to the branch logic in
		// GatherSums.
		GatherSums<NumCounters>(lane, Mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
	} else if(4 == NumBits) {
		GatherSums<NumCounters>(lane, Mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 2>(lane, 0, warpCounters_shared);
	} else if(5 == NumBits) {
		GatherSums<NumCounters>(lane, Mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 2>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 4>(lane, 0, warpCounters_shared);
	} else if(6 >= NumBits) {
		GatherSums<NumCounters>(lane, Mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 2>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 4>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 8>(lane, 0, warpCounters_shared);
	}

	// There are probably still multiple copies of each sum. Perform a parallel
	// scan to add them all up.
	if(NumChannels < WARP_SIZE) {
		volatile uint* reduction = warpCounters_shared + lane;
		uint x = reduction[0];
		#pragma unroll
		for(int i = 0; i < 6 - NumBits; ++i) {
			uint offset = NumChannels<< i;
			if(offset + lane < WARP_SIZE) {
				uint y = reduction[offset];
				x += y;
			}
			reduction[0] = x;
		}
	}
	
	// Re-index the counters so the low short is bucket i and the high short
	// is bucket i + NumBuckets / 2.
	uint packed0, packed1;
	if(1 == NumBits) packed0 = counters_shared[0];
	else if(NumBits <= 6) {
		if(lane < NumChannels) {
			uint halfLane = lane / 2;
			uint low = warpCounters_shared[halfLane];
			uint high = warpCounters_shared[halfLane + NumCounters];
			packed0 = prmt(low, high, (1 & lane) ? 0x7632 : 0x5410);
		}
	} else if(7 == NumBits) {
		uint halfLane = lane / 2;
		uint x = warpCounters_shared[halfLane];
		uint y = warpCounters_shared[halfLane + NumChannels];
		uint z = warpCounters_shared[halfLane + 2 * NumChannels];
		uint w = warpCounters_shared[halfLane + 3 * NumChannels];
		uint mask = (1 & lane) ? 0x7632 : 0x5410;
		packed0 = prmt(x, y, mask);
		packed1 = prmt(z, w, mask);
	}
	__syncthreads();
	
	if(lane < NumChannels)
		counts_shared[lane] = packed0;
	if(7 == NumBits)
		counts_shared[WARP_SIZE + lane] = packed1;
}


////////////////////////////////////////////////////////////////////////////////
// CountFunc

// Have each warp process a histogram for one block. If the block has
// NUM_VALUES, each thread process NUM_VALUES / WARP_SIZE. Eg if NUM_VALUES is
// 2048, each thread process 64 values. This operation is safe for NUM_VALUES up
// to 4096.
template<int NumBits, int NumThreads, int InnerLoop, int Mode>
DEVICE2 void CountFunc(const uint* keys_global, uint bit, uint numElements, 
	uint vt, uint* counts_global) {

	const int NumBuckets = 1<< NumBits;
	const int NumCounters = DIV_UP(NumBuckets, 4);
	const int NumChannels = DIV_UP(NumBuckets, 2);
	
	const int NumWarps = NumThreads / WARP_SIZE;

	const int WarpMem = WARP_SIZE * NumCounters;

	__shared__ volatile uint blockCounters_shared[NumCounters * NumThreads];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint warpStart = (block * NumWarps + warp) * (WARP_SIZE * vt);

	volatile uint* warpCounters_shared = blockCounters_shared + warp * WarpMem;
	volatile uint* counters_shared = warpCounters_shared + lane;

	// Define the counters so we can pass them to IncBucketCounter. They don't
	// actually get used unless NUM_BITS <= 3 however.
	uint counter0 = 0;
	uint counter1 = 0;

	// clear all the counters
	#pragma unroll
	for(int i = 0; i < NumCounters; ++i)
		counters_shared[WARP_SIZE * i] = 0;

	uint values[InnerLoop];
	if(warpStart < numElements) {
		// Unroll to read 8 values at a time. We can also read from uint4!
		const uint* warpData = keys_global + warpStart + lane;
		uint end = vt / InnerLoop;

		// Consume InnerLoop values at a time.
		for(int i = 0; i < end; ++i) {
			
			// Load InnerLoop values from global memory.
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j)
				values[j] = warpData[j * WARP_SIZE];
			warpData += InnerLoop * WARP_SIZE;

			// Extract the digit and increment the bucket counter.
			#pragma unroll
			for(int j = 0; j < InnerLoop; ++j) {
				uint digit = bfe(values[j], bit, NumBits);
				IncBucketCounter(digit, counters_shared, counter0, counter1,
					NumBits);
			}
		}

		// Write the counters to shared memory if they were stored in register.
		if(NumBits <= 3)
			counters_shared[0] = counter0;
		if(3 == NumBits)
			counters_shared[WARP_SIZE] = counter1;
	}
	
	GatherSumsReduce<NumBits, Mode>(warpCounters_shared, lane, 
		blockCounters_shared + NumChannels * warp);

	// Store the counts to global memory.
	uint offset = block * ROUND_UP(NumWarps * NumChannels, WARP_SIZE);
	for(int i = tid; i < NumChannels * NumWarps; i += NumThreads)
		counts_global[offset + i] = blockCounters_shared[i];/**/
}

//template<int NumBits, int Mode>
//DEVICE2 void GatherSumsReduce(volatile uint* warpCounters_shared, uint lane,
//	uint* counts_shared) {


////////////////////////////////////////////////////////////////////////////////

#define GEN_COUNT_FUNC(Name, NumThreads, NumBits, InnerLoop, Mode,			\
	BlocksPerSM)															\
																			\
extern "C" __global__ __launch_bounds__(NumThreads, BlocksPerSM)			\
void Name(const uint* keys_global, uint bit, uint numElements, uint vt,		\
	uint* counts_global) {													\
																			\
	CountFunc<NumBits, NumThreads, InnerLoop, Mode>(						\
		keys_global, bit, numElements, vt, counts_global);					\
}
