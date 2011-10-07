#include "common.cu"

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

	uint targetTemp[ColHeight];

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
	//	if(4 & bucket) counter1 = shl_add(1, shift, counter1);
	//	else counter0 = shl_add(1, shift, counter0);

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

DEVICE int2 TestSortOrder(volatile uint* sorted_scratch, uint value, 
	uint bucket, uint bit, uint numBits, uint lane, uint pass) {

	// Alternate the scratch space so that the last thread's value of the 
	// preceding pass is always available. Store the full value so that we can
	// test the order of both the full key and of the radix digit.
	uint preceding;
	if(1 & pass) {
		sorted_scratch[WARP_SIZE] = value;
		preceding = sorted_scratch[WARP_SIZE - 1];
	} else {
		sorted_scratch[0] = value;
		preceding = sorted_scratch[lane ? -1 : 63];
	}
	uint precedingBucket = bfe(preceding, bit, numBits);

	return make_int2(preceding <= value, precedingBucket <= bucket);
}
