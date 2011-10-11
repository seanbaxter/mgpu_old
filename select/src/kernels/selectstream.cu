

////////////////////////////////////////////////////////////////////////////////
// STREAM PASS

// InitValueStream/InitPairStream adjusts the global pointers and returns a pair
// of counter terms for indexing within the shared memory. 

template<typename T>
int2 InitValueStream(uint lane, T*& values_global, int streamOffset) {

}


template<typename T>
DEVICE2 void PushValue(volatile T* values, T x, bool push, uint lane, uint mask, 
	int2& counters) {

	uint stream = __ballot(push);
	int streamDest = end + __popc(mask & stream);
	end += __popc(stream);

	if(push) values[streamDest] = x;
}

template<typename T>
DEVICE2 void PushPair(volatile T* values, volatile uint* indices, T x,
	uint index, bool push, uint lane, uint mask, int& end) {

	uint stream = __ballot(push);
	int streamDest = end + __popc(mask & stream);
	end += __popc(stream);

	if(push) values[streamDest] = x;
	if(push) indices[streamDest] = index;
}

template<typename T>
DEVICE2 void StreamValues(volatile T* values, uint lane, int& start, int& end, 
	T*& values_global, int streamCount) {

	if(-1 == streamCount) {
		// Stream all values to the target.


	} else if(end > WARP_SIZE * streamCount) {
		int last = ~(WARP_SIZE - 1) & end;

		// 


	}
}


template<typename T>
DEVICE2 void KSmallestStreamValueLoop(const T* source_global, int& offset,
	int end, uint lane, uint warp, uint digitMask, uint digit, int loopCount,
	volatile uint* warpShared, int& valuesStart, int& valuesEnd,
	T*& values_global, int streamCount, uint warpMask, int loopCount) {

	#pragma unroll
	for(int i = 0; i < loopCount; ++i) {
		int source = offset + i * WARP_SIZE + lane;
		T x;
		bool push = false;
		if((-1 != loopCount) || (source < end)) {
			x = source_global[source];
			if(digit == (mask & x)) push = true;
		}

		PushValue(warpShared, x, push, lane, warpMask, valuesEnd);
	}
}


////////////////////////////////////////////////////////////////////////////////



template<typename T>
DEVICE2 void KSmallestStreamValue(const T* source_global, 
	const int2* range_global, const int* streamOffset_global, T* target_global, 
	uint mask, uint digit) {

	// Buffer up to 16 values per thread.
	const int ValuesPerThread = 16;
	const int InnerLoop = 4;

	__shared__ volatile T shared[NUM_THREADS * ValuesPerThread];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile T* warpShared = shared + 16 * WARP_SIZE * warp;

	uint warpMask = bfi(0, 0xffffffff, 0, lane);

	// streamOffset is the first streaming offset for values.
	int streamOffset = streamOffset_global[gid];

	// streamStart is the index of the first lane to write in StreamValues.
	int streamStart = (WARP_SIZE - 1) & streamOffset;
	int streamNext = streamStart;

	// Advance target_global to the beginning of the first memory segment to be
	// addressed by this warp.
	target_global += ~(WARP_SIZE - 1) & streamOffset;
	
	// Round range.y down.
	int end = ROUND_DOWN(range.y, WARP_SIZE * InnerLoop);
	while(range.x < end) {
		
		T x = source_global[range.x + lane];
		uint masked = mask & ConvertToUint(x);
		bool push = masked == digit;

		PushValue(warpShared, x, push, lane, warpMask, streamNext);
						
		
		range.x += InnerLoop * WARP_SIZE;
	}

	// 
	while(range.x < range.y) {
		
	}
	
	StreamValues(warpShared, lane, streamStart, streamNext, target_global, 
		true);
}
	

////////////////////////////////////////////////////////////////////////////////


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamUint(const uint* source_global, const int2* range_global, 
	const int* streamOffset_global, uint* target_global, uint mask,
	uint digit) {

	KSmallestStream(source_global, range_global, streamOffset_global,
		target_global, mask, digit);
}
	
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamInt(const int* source_global, const int2* range_global, 
	const int* streamOffset_global, int* target_global, uint mask, uint digit) {

	KSmallestStream(source_global, range_global, streamOffset_global, 
		target_global, mask, digit);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamFloat(const float* source_global, const int2* range_global, 
	const int* streamOffset_global, float* target_global, uint mask, 
	uint digit) {

	KSmallestStream(source_global, range_global, streamOffset_global,
		target_global, mask, digit);
}

