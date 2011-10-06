#include "common.cu"

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
