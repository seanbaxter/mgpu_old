#include "common.cu"


////////////////////////////////////////////////////////////////////////////////
// GatherSums is copied from sort/src/kernels/countcommon.cu

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



////////////////////////////////////////////////////////////////////////////////
// COUNT PASS

DEVICE void IncCounter(volatile uint* counters, uint bucket) {
	uint index = (32 / 4) * (~3 & bucket);
	uint counter = counters[index];
	counter = shl_add(1, bfi(0, bucket, 3, 2), counter);
	counters[index] = counter;
}

DEVICE void ClearCounters(volatile uint* counters) {
	#pragma unroll
	for(int i = 0; i < 16; ++i)
		counters[i * WARP_SIZE] = 0;
}

DEVICE void ExpandCounters(uint lane, volatile uint* warpCounters,
	uint threadTotals[2]) {

	GatherSums<16>(lane, 2, warpCounters);
	GatherSums<16>(lane, 0, warpCounters);
	GatherSums<8>(lane, 0, warpCounters);
	GatherSums<4>(lane, 0, warpCounters);
	GatherSums<2>(lane, 0, warpCounters);

	// Unpack even and odd counters.
	uint x = warpCounters[lane];
	threadTotals[0] += 0xffff & x;
	threadTotals[1] += x>> 16;

	ClearCounters(warpCounters + lane);
}
