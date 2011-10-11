

////////////////////////////////////////////////////////////////////////////////
// STREAM PASS

////////////////////////////////////////////////////////////////////////////////
// InitValueStream/InitPairStream adjusts the global pointers and returns a pair
// of counter terms for indexing within the shared memory. .x is the offset of 
// the first lane with a cached value/index. .y is the end of the cached array
// (one past the last element). To ensure coalesced stores, the pointer is
// rounded down to a segment multiple and the counters.x term (the stream cache
// start) is set to the offset within the segment. This way each thread stores
// only to its own lane. After the first store, counters.x is cleared.

DEVICE int2 InitValueStream(uint lane, uint*& values_global, int streamOffset) {
	// Add streamOffset (the index for this warp's first store) and round down
	// to a segment multiple.
	values_global += ~(WARP_SIZE - 1) & streamOffset;
	int start = (WARP_SIZE - 1) & streamOffset;

	// The start and end terms begin the same, as they array is empty.
	return make_int2(start, start);
}

DEVICE int2 InitPairStream(uint lane, uint*& values_global,
	uint*& indices_global, int streamOffset) {

	int offset = ~(WARP_SIZE - 1) & streamOffset;
	values_global += offset;
	indices_global += offset;

	int start = (WARP_SIZE - 1) & streamOffset;
	return make_int2(start, start);
}


////////////////////////////////////////////////////////////////////////////////
// PushValue/PushPair performs an in-order inserted of a value or pair into
// shared memory. We may be able to save an instruction by always storing the
// value or index to shared memory. This is not a risk, as PushValue/PushPair
// is only called when there is enough available space to accommodate an entire
// warp of data.

DEVICE void PushValue(volatile uint* values, uint x, bool push, uint warpMask,
	int2& counters) {

	uint stream = __ballot(push);
	int streamDest = counters.y + __popc(warpMask & stream);
	counters.y += __popc(stream);

	// if(push)
		values[streamDest] = x;
}

DEVICE void PushPair(volatile uint* values, volatile uint* indices, uint x,
	uint index, bool push, uint warpMask, int2& counters) {

	uint stream = __ballot(push);
	int streamDest = counters.y + __popc(warpMask & stream);
	counters.y += __popc(stream);

	// if(push)
		values[streamDest] = x;
	// if(push)
		indices[streamDest] = index;
}


////////////////////////////////////////////////////////////////////////////////
// StreamValues/Stream

DEVICE2 void StreamValues(volatile uint* values, uint lane, int2& counters,
	uint*& values_global, int streamCount) {

	if(-1 == streamCount) {
		// Stream all values to the target. This is called after the end of
		// each warp's range. Keep it simple by performang a range check each
		// iteration.
		for(int i = lane; i < counters.y; i += WARP_SIZE)
			if((i >= counters.x) && (i < counters.y))
				values_global[i] = values[i];

		// Adjust the counters to point to the position after the last element
		// stored. These are likely not generated, as StreamValues with a -1
		// streamCount is only called at the end of the kernel (except when
		// debugging).
		values_global += ROUND_DOWN(counters.y, WARP_SIZE);
		counters.x = (WARP_SIZE - 1) & counters.y;
		counters.y = counters.x;

	} else if(counters.y > streamCount * WARP_SIZE) {
		// Stream only the first and all the complete segments. The tail values
		// are compacted back to the front to give coalesced stores. This is 
		// called in the main loop for the warp's range.

		// Store the first segment, which is possibly fractional.
		if(lane >= counters.x)
			values_global[lane] = values[lane];

		// Store the remaining complete segments.
		#pragma unroll
		for(int i = 1; i < streamCount; ++i)
			values_global[i * WARP_SIZE + lane] = values[i * WARP_SIZE + lane];

		// Move the fractional last segment to the front.
		values[lane] = values[streamCount * WARP_SIZE + lane];

		counters.x = 0;
		counters.y -= WARP_SIZE * streamCount;;
		values_global += streamCount * WARP_SIZE;
	}
}


// Pass gid for debugging.
template<typename T>
DEVICE2 void KSmallestStreamValueIt(const T* source_global, uint gid, int& offset,
	int end, uint digitMask, uint digit, volatile uint* warpShared, 
	int2& counters, bool check, uint warpMask) {

	// Set x to an invalid value. If the offset is in the warp's range, load
	// the source data, convert it to radix order, and mask out the bits of
	// interest.
	uint val;
	uint x = 0xffffffff;
	if(!check || (offset < end)) {
		val = source_global[offset];
		x = digitMask & ConvertToUint(val);
	}

	// Push this iteration's values.
	PushValue(warpShared, x, x == digit, warpMask, counters);

	offset += WARP_SIZE;
}



// NOTE: have to manually unroll loop here because nvcc gives 
//		Advisory: Loop was not unrolled, unexpected call OPs
// error when calling __ballot from an unrolled loop.
template<typename T, int LoopCount>
DEVICE2 void KSmallestStreamValueLoop(const T* source_global, uint gid, bool check,
	int& offset, int end, uint lane, uint digitMask, uint digit,
	volatile uint* warpShared, int2& counters, uint*& values_global,
	uint warpMask) {

	KSmallestStreamValueIt(source_global, gid, offset, end, digitMask, digit,
		warpShared, counters, check, warpMask);

	if(LoopCount >= 2)
		KSmallestStreamValueIt(source_global, gid, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);

	if(LoopCount >= 4) {
		KSmallestStreamValueIt(source_global, gid, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
		KSmallestStreamValueIt(source_global, gid, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
	}
}


////////////////////////////////////////////////////////////////////////////////


template<typename T>
DEVICE2 void KSmallestStreamValue(const T* source_global, 
	const int2* range_global, const int* streamOffset_global, 
	uint* values_global, uint digitMask, uint digit, uint* debug_global) {

	// Buffer up to 16 values per thread.
	const int ValuesPerThread = 16;
	const int InnerLoop = 4;

	__shared__ volatile uint shared[NUM_THREADS * ValuesPerThread];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile uint* warpShared = shared + 16 * WARP_SIZE * warp;

	uint warpMask = bfi(0, 0xffffffff, 0, lane);

	// streamOffset is the first streaming offset for values.
	int streamOffset = streamOffset_global[gid];

	int2 counters = InitValueStream(lane, values_global, streamOffset);

	debug_global[2 * (32 * gid + lane)] = counters.x;
	range.x += lane;

	// Round range.y down.
	int end = ROUND_DOWN(range.y, WARP_SIZE * InnerLoop);
	while(range.x < end) {
		KSmallestStreamValueLoop<T, InnerLoop>(source_global, gid, false, range.x, end, 
			lane, digitMask, digit, warpShared, counters, values_global, 
			warpMask);
		StreamValues(warpShared, lane, counters, values_global, InnerLoop);
	}

	end = ROUND_UP(range.y, WARP_SIZE) + lane;
	while(range.x < end)
		// Process the end of the array.
		KSmallestStreamValueLoop<T, 1>(source_global, gid, true, range.x, range.y,
			lane, digitMask, digit, warpShared, counters, values_global, 
			warpMask);

	StreamValues(warpShared, lane, counters, values_global, -1);

	debug_global[2 * (32 * gid + lane) + 1] = counters.y;
}



////////////////////////////////////////////////////////////////////////////////


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamUintValue(const uint* source_global,
	const int2* range_global, const int* streamOffset_global,
	uint* target_global, uint mask, uint digit, uint* debug_global) {

	KSmallestStreamValue(source_global, range_global, streamOffset_global,
		target_global, mask, digit, debug_global);
}
	
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamIntValue(const int* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask,
	uint digit, uint* debug_global) {

	KSmallestStreamValue(source_global, range_global, streamOffset_global, 
		target_global, mask, digit, debug_global);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamFloat(const float* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask, 
	uint digit, uint* debug_global) {

	KSmallestStreamValue(source_global, range_global, streamOffset_global,
		target_global, mask, digit, debug_global);
}

