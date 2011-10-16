

////////////////////////////////////////////////////////////////////////////////
// STREAM PASS

////////////////////////////////////////////////////////////////////////////////
// InitKeyStream/InitPairStream adjusts the global pointers and returns a pair
// of counter terms for indexing within the shared memory. .x is the offset of 
// the first lane with a cached key/index. .y is the end of the cached array
// (one past the last element). To ensure coalesced stores, the pointer is
// rounded down to a segment multiple and the counters.x term (the stream cache
// start) is set to the offset within the segment. This way each thread stores
// only to its own lane. After the first store, counters.x is cleared.

DEVICE int2 InitKeyStream(uint lane, uint*& keys_global, int streamOffset) {
	// Add streamOffset (the index for this warp's first store) and round down
	// to a segment multiple.
	keys_global += ~(WARP_SIZE - 1) & streamOffset;
	int start = (WARP_SIZE - 1) & streamOffset;

	// The start and end terms begin the same, as they array is empty.
	return make_int2(start, start);
}

DEVICE int2 InitPairStream(uint lane, uint*& keys_global, uint*& indices_global,
	int streamOffset) {

	int offset = ~(WARP_SIZE - 1) & streamOffset;
	keys_global += offset;
	indices_global += offset;

	int start = (WARP_SIZE - 1) & streamOffset;
	return make_int2(start, start);
}


////////////////////////////////////////////////////////////////////////////////
// PushValue/PushPair performs an in-order inserted of a key or pair into
// shared memory. We may be able to save an instruction by always storing the
// key or index to shared memory. This is not a risk, as PushValue/PushPair
// is only called when there is enough available space to accommodate an entire
// warp of data.

DEVICE void PushValue(volatile uint* keys, uint x, bool push, uint warpMask,
	int2& counters) {

	uint stream = __ballot(push);
	int streamDest = counters.y + __popc(warpMask & stream);
	counters.y += __popc(stream);

	// if(push)
		keys[streamDest] = x;
}

DEVICE void PushPair(volatile uint* keys, volatile uint* indices, uint x,
	uint index, bool push, uint warpMask, int2& counters) {

	uint stream = __ballot(push);
	int streamDest = counters.y + __popc(warpMask & stream);
	counters.y += __popc(stream);

	// if(push)
		keys[streamDest] = x;
	// if(push)
		indices[streamDest] = index;
}


////////////////////////////////////////////////////////////////////////////////
// StreamValues/Stream

DEVICE2 void StreamKeys(volatile uint* keys, uint lane, int2& counters,
	uint*& keys_global, int streamCount) {

	if(-1 == streamCount) {
		// Stream all keys to the target. This is called after the end of
		// each warp's range. Keep it simple by performang a range check each
		// iteration.
		for(int i = lane; i < counters.y; i += WARP_SIZE)
			if((i >= counters.x) && (i < counters.y))
				keys_global[i] = keys[i];

		// Adjust the counters to point to the position after the last element
		// stored. These are likely not generated, as StreamValues with a -1
		// streamCount is only called at the end of the kernel (except when
		// debugging).
		keys_global += ROUND_DOWN(counters.y, WARP_SIZE);
		counters.x = (WARP_SIZE - 1) & counters.y;
		counters.y = counters.x;

	} else if(counters.y > streamCount * WARP_SIZE) {
		// Stream only the first and all the complete segments. The tail keys
		// are compacted back to the front to give coalesced stores. This is 
		// called in the main loop for the warp's range.

		// Store the first segment, which is possibly fractional.
		if(lane >= counters.x)
			keys_global[lane] = keys[lane];

		// Store the remaining complete segments.
		#pragma unroll
		for(int i = 1; i < streamCount; ++i)
			keys_global[i * WARP_SIZE + lane] = keys[i * WARP_SIZE + lane];

		// Move the fractional last segment to the front.
		keys[lane] = keys[streamCount * WARP_SIZE + lane];

		counters.x = 0;
		counters.y -= WARP_SIZE * streamCount;;
		keys_global += streamCount * WARP_SIZE;
	}
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// SINGLE BUCKET, NO INDEX

template<typename T>
DEVICE2 void SelectStreamAIt(const T* source_global, int& offset, int end,
	uint digitMask, uint digit, volatile uint* warpShared, int2& counters,
	bool check, uint warpMask) {

	// Set x to an invalid digit. If the offset is in the warp's range, load
	// the source data, convert it to radix order, and mask out the bits of
	// interest.
	uint val;
	uint x = 0xffffffff;
	if(!check || (offset < end)) {
		T val2 = source_global[offset];
		val = ReinterpretToUint(val2);
		x = digitMask & ConvertToUint(val2);
	}

	// Push this iteration's keys.
	PushValue(warpShared, val, x == digit, warpMask, counters);

	offset += WARP_SIZE;
}


// NOTE: have to manually unroll loop here because nvcc gives 
//		Advisory: Loop was not unrolled, unexpected call OPs
// error when calling __ballot from an unrolled loop.
template<typename T, int LoopCount>
DEVICE2 void SelectStreamALoop(const T* source_global, bool check,
	int& offset, int end, uint lane, uint digitMask, uint digit,
	volatile uint* warpShared, int2& counters, uint*& keys_global,
	uint warpMask) {

	SelectStreamAIt(source_global, offset, end, digitMask, digit, warpShared, 
		counters, check, warpMask);

	if(LoopCount >= 2)
		SelectStreamAIt(source_global, offset, end, digitMask, digit, 
			warpShared, counters, check, warpMask);

	if(LoopCount >= 4) {
		SelectStreamAIt(source_global, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
		SelectStreamAIt(source_global, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
	}
}


////////////////////////////////////////////////////////////////////////////////
// SelectStreamValue
// Implements single-bucket stream with no indexing.

template<typename T>
DEVICE2 void SelectStreamA(const T* source_global, 
	const int2* range_global, const int* streamOffset_global, 
	uint* keys_global, uint digitMask, uint digit, uint* debug_global) {

	// Buffer up to 16 keys per thread.
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

	// streamOffset is the first streaming offset for keys.
	int streamOffset = streamOffset_global[gid];
	int2 counters = InitKeyStream(lane, keys_global, streamOffset);

	range.x += lane;

	// Round range.y down.
	int end = ROUND_DOWN(range.y, WARP_SIZE * InnerLoop);
	while(range.x < end) {
		SelectStreamALoop<T, InnerLoop>(source_global, false, range.x,
			end, lane, digitMask, digit, warpShared, counters, keys_global, 
			warpMask);
		StreamKeys(warpShared, lane, counters, keys_global, InnerLoop);
	}

	end = ROUND_UP(range.y, WARP_SIZE) + lane;
	while(range.x < end)
		// Process the end of the array.
		SelectStreamALoop<T, 1>(source_global, true, range.x, range.y, lane,
			digitMask, digit, warpShared, counters, keys_global, warpMask);

	StreamKeys(warpShared, lane, counters, keys_global, -1);
}


////////////////////////////////////////////////////////////////////////////////
// CUDA Kernels

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void SelectStreamUintA(const uint* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask, uint digit,
	uint* debug_global) {

	SelectStreamA(source_global, range_global, streamOffset_global, 
		target_global, mask, digit, debug_global);
}
	
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void SelectStreamIntA(const int* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask, uint digit,
	uint* debug_global) {

	SelectStreamA(source_global, range_global, streamOffset_global, 
		target_global, mask, digit, debug_global);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void SelectStreamFloatA(const float* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask, uint digit,
	uint* debug_global) {

	SelectStreamA(source_global, range_global, streamOffset_global,
		target_global, mask, digit, debug_global);
}



/*
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// SINGLE BUCKET, GENERATE INDICES


template<typename T>
DEVICE2 void SelectStreamValueIt(const T* source_global, int& offset,
	int end, uint digitMask, uint digit, volatile uint* warpShared, 
	int2& counters, bool check, uint warpMask) {

	// Set x to an invalid key. If the offset is in the warp's range, load
	// the source data, convert it to radix order, and mask out the bits of
	// interest.
	uint val;
	uint x = 0xffffffff;
	if(!check || (offset < end)) {
		val = source_global[offset];
		x = digitMask & ConvertToUint(val);
	}

	// Push this iteration's keys.
	PushValue(warpShared, x, x == digit, warpMask, counters);

	offset += WARP_SIZE;
}


// NOTE: have to manually unroll loop here because nvcc gives 
//		Advisory: Loop was not unrolled, unexpected call OPs
// error when calling __ballot from an unrolled loop.
template<typename T, int LoopCount>
DEVICE2 void SelectStreamValueLoop(const T* source_global, bool check,
	int& offset, int end, uint lane, uint digitMask, uint digit,
	volatile uint* warpShared, int2& counters, uint*& keys_global,
	uint warpMask) {

	SelectStreamValueIt(source_global, offset, end, digitMask, digit,
		warpShared, counters, check, warpMask);

	if(LoopCount >= 2)
		SelectStreamValueIt(source_global, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);

	if(LoopCount >= 4) {
		SelectStreamValueIt(source_global, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
		SelectStreamValueIt(source_global, offset, end, digitMask, digit,
			warpShared, counters, check, warpMask);
	}
}


*/