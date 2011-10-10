

////////////////////////////////////////////////////////////////////////////////
// STREAM PASS

template<typename T>
DEVICE2 void PushValue(volatile T* values, T x, bool push, uint lane, uint mask, 
	int& end) {

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



#if 0

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
						
		
		range.x += INNER_LOOP_STREAM * WARP_SIZE;
	}

	// 
	while(range.x < range.y) {
		
	}
	
	StreamValues(warpShared, lane, streamStart, streamNext, target_global, 
		true);
}
	


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




__global__ int ksmallestStream_global;
__global__ int ksmallestLeft_global;

#define NUM_THREADS 128
#define NUM_WARPS 4
#define VALUES_PER_THREAD 5
#define VALUES_PER_WARP (VALUES_PER_THREAD * WARP_SIZE)

// Reserve 2 * WARP_SIZE values per warp. As soon as WARP_SIZE values are
// available per warp, store them to device memory.
__shared__ volatile uint values_shared[NUM_THREADS * (VALUES_PER_THREAD + 1)];
__shared__ volatile uint indices_shared[NUM_THREADS * (VALUES_PER_THREAD + 1)];

DEVICE2 uint ConvertToUint(uint x) { return x; }
DEVICE2 uint ConvertToUint(int x) { return (uint)x; }
DEVICE2 uint ConvertToUint(float x) { return (uint)__float_as_int(x); }

// Define absolute min and absolute max.

////////////////////////////////////////////////////////////////////////////////
// CompareAndStream

DEVICE bool StreamToGlobal(uint* dest_global, uint* indices_global, int lane,
	int& count, bool writeAll, int capacity, volatile uint* valuesShared, 
	volatile uint* indicesShared) { 

	int valuesToWrite = count;
	if(!writeAll) valuesToWrite = ~(WARP_SIZE - 1) & count;
	count -= valuesToWrite;

	// To avoid atomic resource contention, have just one thread perform the
	// interlocked operation.
	volatile uint* advance_shared = valuesShared + VALUES_PER_WARP - 1;
	if(!lane) 
		*advance_shared = atomicAdd(&ksmallestStream_global, valuesToWrite);
	int target = *advance_shared;
	if(target >= capacity) return false;

	target += lane;
	valuesToWrite -= lane;

	uint source = lane;
	while(valuesToWrite >= 0) {
		dest_global[target] = valuesShared[source];
		indices_global[target] = indicesShared[source];
		source += WARP_SIZE;
		valuesToWrite -= WARP_SIZE;
	}

	// Copy the values form the end of the shared memory array to the front.
	if(count > lane) {
		valuesShared[lane] = valuesShared[source];
		indicesShared[lane] = indicesShared[source];
	}
	return true;
}


DEVICE2 template<typename T>
void ProcessStream(const T* source_global, uint* dest_global,
	uint* indices_global, int2 range, T left, T right, int capacity, 
	bool checkRange, uint lane, uint warp) {

	uint leftCounter = 0;
	int warpSharedPop = 0;
	uint mask = bfi(0, 0xffffffff, 0, tid);

	volatile uint* valuesShared = values_shared + warp * VALUES_PER_WARP;
	volatile uint* indicesShared = indices_shared + warp * VALUES_PER_WARP;

	while(warpRange.x < warpRange.y) {
		uint source = warpRange.x + lane;
		
		T val = source_global[source];

		// There are three cases:
		// 1) The value comes before the left splitter. For this, increment
		// leftCounter.
		// 2) The value comes after the right spliter. For this, do nothing.
		// 3) The value comes between the splitters. For this, move the value
		// to shared memory and periodically stream to global memory.
		
		bool inRange = false;
		if(val < left) ++leftCounter;
		else if(val <= right) inRange = true;

		uint warpValues = __ballot(inRange);

		// Mask out the values at and above tid to find the offset.
		uint offset = __popc(mask & warpValues) + warpSharedPop;
		uint advance = __popc(warpValues);
		warpSharedPop += advance;

		if(inRange) valuesShared[offset] = ConvertToUint(val);
		if(inRange) indicesShared[offset] = source;

		if(warpSharedPop >= VALUES_PER_WARP) {
			bool success = StreamToGlobal(dest_global, indices_global, lane,
				warpSharedPop, false, capacity, valuesShared, indicesShared);
			if(!success) return;
		}

		warpRange.x += WARP_SIZE;	
	}

	// Sum up the number of left counters.
	valuesShared[lane] = leftCounter;
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		if(lane >= offset) leftCounter += valuesShared[lane - offset];
		valuesShared[lane] = leftCounter;
	}

	if(WARP_SIZE - 1 == lane)
		atomicAdd(&ksmallestLeft_global, leftCounter);
}

template<typename T>
DEVICE2 void CompareAndStream(const T* source_global, uint* dest_global,
	uint* indices_global, const int2* range_global, T left, T right, 
	int capacity) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint gid = blockIdx.x * NUM_WARPS + warp;

	int2 warpRange = range_global[gid];


	if(gid < NUM_WARPS * blockDim.x) {
		ProcessStream<T>(source_global, dest_global, warpRange, 
			left, right, capacity, false, tid, lane, warp);
	} else {
		ProcessStream<T>(source_global, dest_global, warpRange, 
			left, right, capacity, true, tid, lane, warp);
	}
}


#endif // 0
