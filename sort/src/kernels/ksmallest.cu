#include "countcommon.cu"


// Use 128 threads and 6 blocks per SM for 50% occupancy.
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define BLOCKS_PER_SM 6

#define INNER_LOOP 8

__shared__ volatile uint counts_shared[NUM_THREADS * 16];

DEVICE2 uint ConvertToUint(uint x) { return x; }
DEVICE2 uint ConvertToUint(int x) { return x + 0x80000000; }
DEVICE2 uint ConvertToUint(float x) { return (uint)__float_as_int(x); }


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


template<typename T>
DEVICE2 void KSmallestCount(const T* source_global, uint* hist_global, 
	const int2* range_global, uint shift) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile uint* warpShared = counts_shared + 16 * WARP_SIZE * warp;
	volatile uint* tidShared = warpShared + lane;

	ClearCounters(tidShared);

	// Each warp holds two unpacked totals.
	uint threadTotals[2] = { 0, 0 };
	int dumpTime = 0;

	while(range.x < range.y) {
		
		#pragma unroll
		for(int i = 0; i < INNER_LOOP; ++i) {
			T x = source_global[range.x + i * WARP_SIZE + lane];
			uint digit = bfe(ConvertToUint(x), shift, 6);
			IncCounter(tidShared, digit);
		}
		dumpTime += INNER_LOOP;
		range.x += INNER_LOOP * WARP_SIZE;

		if(dumpTime >= (256 - INNER_LOOP)) {
			ExpandCounters(lane, warpShared, threadTotals);
			dumpTime = 0;
		}
	}
	ExpandCounters(lane, warpShared, threadTotals);

	hist_global[64 * gid + 2 * lane] = threadTotals[0];
	hist_global[64 * gid + 2 * lane + 1] = threadTotals[1];
}


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountUint(const uint* source_global, uint* hist_global,
	const int2* range_global, uint shift) {

	KSmallestCount(source_global, hist_global, range_global, shift);

}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountInt(const int* source_global, uint* hist_global,
	const int2* range_global, uint shift) {

	KSmallestCount(source_global, hist_global, range_global, shift);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountFloat(const float* source_global, uint* hist_global,
	const int2* range_global, uint shift) {

	KSmallestCount(source_global, hist_global, range_global, shift);
}


////////////////////////////////////////////////////////////////////////////////
// HISTOGRAM PASS

//extern "C" __global__
//void 



////////////////////////////////////////////////////////////////////////////////
// STREAM PASS


template<typename T>
DEVICE2 void KSmallestStream(const T* source_global, const uint* streamOffset,
	const int2* range_global, uint* target_global, uint mask, uint digit) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile uint* warpShared = counts_shared + 16 * WARP_SIZE * warp;
	volatile uint* tidShared = warpShared + lane;

	uint warpMask = bfi(0, 0xffffffff, 0, lane);

	while(range.x < range.y) {
		
		#pragma unroll
		for(int i = 0; i < INNER_LOOP; ++i) {
			T x = source_global[range.x + i * WARP_SIZE + lane];
			uint masked = mask & ConvertToUint(x);
			bool stream = masked == digit;
						
			uint warpStream = __ballot(stream);
			uint streamOffset = __popc(warpMask & warpStream);
			uint streamAdvance = __popc(warpStream);

			if(stream) {

			}

			// If 

		}
		range.x += INNER_LOOP * WARP_SIZE;
	}

}
	
	


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamUint(const uint* source_global, const uint* streamOffset,
	const int2* range_global, uint* target_global, uint mask, uint digit) {

	KSmallestStream(source_global, streamOffset, range_global, target_global,
		mask, digit);
}
	
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamInt(const int* source_global, const uint* streamOffset,
	const int2* range_global, uint* target_global, uint mask, uint digit) {

	KSmallestStream(source_global, streamOffset, range_global, target_global,
		mask, digit);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) 
void KSmallestStreamFloat(const float* source_global, const uint* streamOffset,
	const int2* range_global, uint* target_global, uint mask, uint digit) {

	KSmallestStream(source_global, streamOffset, range_global, target_global,
		mask, digit);
}












#if 0

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
