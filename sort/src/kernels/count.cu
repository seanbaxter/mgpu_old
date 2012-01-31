
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

/*
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
template<int NumBits>
DEVICE2 void GatherSumsReduce(volatile uint* warpCounters_shared, uint lane,
	int mode, volatile uint* counts_shared) {

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
		GatherSums<NumCounters>(lane, mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
	} else if(4 == NumBits) {
		GatherSums<NumCounters>(lane, mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 2>(lane, 0, warpCounters_shared);
	} else if(5 == NumBits) {
		GatherSums<NumCounters>(lane, mode, warpCounters_shared);
		GatherSums<NumCounters>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 2>(lane, 0, warpCounters_shared);
		GatherSums<NumCounters / 4>(lane, 0, warpCounters_shared);
	} else if(6 <= NumBits) {
		GatherSums<NumCounters>(lane, mode, warpCounters_shared);
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
		// lane has values 4 * lane through 4 * lane + 3. We want it to manage
		// values lane, 32 + lane, 64 + lane, and 96 + lane.

		// When serializing, counts 0 through 63 are in bits 0:15. Counts 64
		// through 127 are in bits 16:31.

		uint offset = lane / 4;
		if(1 & (lane / 2)) offset += WARP_SIZE;
		
		uint mask = (1 & lane) ? 0x7632 : 0x5410;
		uint x = warpCounters_shared[offset];			// 0 + lane
		uint y = warpCounters_shared[offset + 8];		// 32 + lane
		uint z = warpCounters_shared[offset + 16];		// 64 + lane
		uint w = warpCounters_shared[offset + 24];		// 96 + lane

		// packed0 = (0 + lane, 64 + lane)
		// packed1 = (32 + lane, 96 + lane)
		packed0 = prmt(x, z, mask);
		packed1 = prmt(y, w, mask);
	}
	__syncthreads();
	
	if(lane < NumChannels)
		counts_shared[lane] = packed0;
	if(7 == NumBits)
		counts_shared[WARP_SIZE + lane] = packed1;
}*/


////////////////////////////////////////////////////////////////////////////////
// GatherSumsReduce

// If mode is 1, we can add byte-packed counters together before unpacking with
// prmt. This is safe for threads that process up to 64 values each (i.e.
// sort blocks with up to 2048 values/block). When mode 2 is asserted, counters
// are first unpacked then added to avoid overflow. We prefer mode 1 for
// performance. Do not pass mode if it is a runtime parameter - in this case,
// revert to mode = 2.

template<int NumBits>
DEVICE2 void GatherSumsReduce(volatile uint* warpCounters_shared, uint lane, 
	int mode, volatile uint* counts_shared) {

	const int NumDigits = 1<< NumBits;
	const int NumChannels = DIV_UP(NumDigits, 2);
	const int NumCounters = DIV_UP(NumDigits, 4);
	const int LanesPerCounter = WARP_SIZE / NumCounters;
	const int SegLength = WARP_SIZE / LanesPerCounter;

	////////////////////////////////////////////////////////////////////////////
	// Reduce the total number of counter rows to 1 or 2.

	if(2 == NumBits) {
		// Grab even and odd counters, and add and widen.
		uint a = warpCounters_shared[~1 & lane];
		uint b = warpCounters_shared[1 | lane];

		// Separate into even and odd bytes.
		uint gather = (1 & lane) ? 0x4341 : 0x4240;
		uint sum = prmt(a, 0, gather) + prmt(b, 0, gather);
		warpCounters_shared[lane] = sum;
	} else if(NumBits >= 3) {
		// If we have at least 2 rows of data, run a sequential reduction by 
		// assigning each thread to read packed digit counters starting from its
		// own lane, and cylically moving along rows. 
		
		// For 3 bits (8 digits - 2 counters), there are 16 threads assigned to
		// each row, and the nominal segment length is two items. That is, lane
		// 0 reads from (0, 0), lane 1 reads from (1, 1), lane 2 reads from
		// (0, 2), lane 3 reads from (1, 3), etc.

		// For 7 bits (128 digits - 32 counters), there is only one thread 
		// assigned to each row, and the segment length is 32 (that is, read 
		// across the entire row). The threads start from different columns to
		// avoid bank conflicts.
		uint lowAccum = 0;
		uint highAccum = 0;

		uint row = (NumCounters - 1) & lane;
		volatile uint* row_shared = warpCounters_shared + row * WARP_SIZE;

		if(1 == mode) {
			#pragma unroll
			for(int i = 0; i < SegLength; i += 2) {
		//		uint offset1 = (4 * WARP_SIZE - 1) & (4 * (lane + i));
		//		uint offset2 = (4 * WARP_SIZE - 1) & (4 * (lane + 1 + i));
		//		uint packed1 = LoadShifted(row_shared, offset1);
		//		uint packed2 = LoadShifted(row_shared, offset2);

				// To eliminate a SHL each iteration, pre-multiply the offsets
				// by 4:
				uint offset1 = (WARP_SIZE - 1) & (lane + i);
				uint offset2 = (WARP_SIZE - 1) & (lane + 1 + i);
				uint packed1 = row_shared[offset1];
				uint packed2 = row_shared[offset2];

				packed1 += packed2;
				uint low = prmt(packed1, 0, 0x4140);
				uint high = prmt(packed1, 0, 0x4342);
				lowAccum += low;
				highAccum += high;
			}
		} else if (2 == mode) {
			#pragma unroll
			for(int i = 0; i < SegLength; ++i) {
				// uint offset = (WARP_SIZE - 1) & (lane + i);
				// uint packed = row_shared[offset];

				// To eliminate a SHL each iteration, pre-multiply the offsets
				// by 4:
				uint offset = (4 * WARP_SIZE - 1) & (4 * (lane + i));
				uint packed = LoadShifted(row_shared, offset);

				uint low = prmt(packed, 0, 0x4140);
				uint high = prmt(packed, 0, 0x4342);
				lowAccum += low;
				highAccum += high;
			}
		}

		warpCounters_shared[lane] = lowAccum;
		warpCounters_shared[WARP_SIZE + lane] = highAccum;
	}


	////////////////////////////////////////////////////////////////////////////
	// Reduce the digit counts to single values.

	if(NumBits <= 2) {
		// Reduce over a single row.
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
	} else {
		// Reduce over two rows.
		volatile uint* reduction = warpCounters_shared + lane;
		uint x0 = reduction[0];
		uint x1 = reduction[WARP_SIZE];
		#pragma unroll
		for(int i = 0; i < 7 - NumBits; ++i) {
			uint offset = (NumChannels / 2)<< i;
			if(offset + lane < WARP_SIZE) {
				uint y0 = reduction[offset];
				uint y1 = reduction[WARP_SIZE + offset];
				x0 += y0;
				x1 += y1;
			}
			reduction[0] = x0;
			reduction[WARP_SIZE] = x1;			
		}
	}


	////////////////////////////////////////////////////////////////////////////
	// Permute the digit counts around so we can serialize key counts like this:
	// 0 1 2 3 4 5 6 7
	// 8 9 a b c d e f,
	// where counts 0-7 are in bits 15:0 and counts 8-f are in bits 31:16.

	if(NumBits <= 2) {
		uint x = warpCounters_shared[lane];
		__syncthreads();
		if(lane < NumChannels) counts_shared[lane] = x;
	} else if(NumBits <= 6) {
		uint quarterLane = lane / 4;
		uint rowOffset = (2 & lane) ? WARP_SIZE : 0;
		uint mask = (1 & lane) ? 0x7632 : 0x5410;

		uint x0 = warpCounters_shared[rowOffset + quarterLane];
		uint x1 = warpCounters_shared[rowOffset + quarterLane + NumCounters / 2];
		uint packed = prmt(x0, x1, mask);
		__syncthreads();

		counts_shared[lane] = packed;
	} else if(7 == NumBits) {
		uint halfLane = lane / 2;
		uint rowOffset = (1 & lane) ? WARP_SIZE : 0;
		
		// For lane 0: x0 = (0, 1), x1 = (64, 65).
		// For lane 1: x0 = (2, 3), x1 = (66, 67).
		uint x0 = warpCounters_shared[rowOffset + halfLane];
		uint x1 = warpCounters_shared[rowOffset + halfLane + 16];

		// For lane 0: low = (0, 64), high = (1, 65).
		// For lane 1: low = (2, 66), high = (3, 67). 
		uint low = prmt(x0, x1, 0x5410);
		uint high = prmt(x0, x1, 0x7632);

		// Store to the counts_shared array in the requested order. This incurs
		// a 2-way bank conflict for each store.
		__syncthreads();
		counts_shared[2 * lane + 0] = low;
		counts_shared[2 * lane + 1] = high;
	}
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

			/*
			#pragma unroll
			for(int j = 0; j < InnerLoop / 4; ++j) {
				const uint4* packedData = (const uint4*)warpData;
				uint4 packed = packedData[j];
				values[4 * j + 0] = packed.x;
				values[4 * j + 1] = packed.y;
				values[4 * j + 2] = packed.z;
				values[4 * j + 3] = packed.w;
			}*/

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

	GatherSumsReduce<NumBits>(warpCounters_shared, lane, Mode,
		blockCounters_shared + NumChannels * warp);

	// Store the counts to global memory.
	uint offset = block * ROUND_UP(NumWarps * NumChannels, WARP_SIZE);
	for(int i = tid; i < NumChannels * NumWarps; i += NumThreads)
		counts_global[offset + i] = blockCounters_shared[i];
}


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
