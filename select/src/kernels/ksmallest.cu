#include "common.cu"


// Use 128 threads and 6 blocks per SM for 50% occupancy.
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define BLOCKS_PER_SM 6

#define INNER_LOOP_COUNT 8
#define INNER_LOOP_STREAM 4


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
DEVICE2 void KSmallestCountItem(const T* source_global, uint lane, int& offset,
	int end, uint shift, uint bits, int loopCount, volatile uint* warpShared,
	uint threadTotals[2], bool check, int& dumpTime, bool forceExpand) {

	#pragma unroll
	for(int i = 0; i < loopCount; ++i) {
		int source = offset + i * WARP_SIZE + lane;
		if(!check || (source < end)) {
			T x = source_global[offset + i * WARP_SIZE + lane];
			uint digit = bfe(ConvertToUint(x), shift, bits);
			IncCounter(warpShared + lane, digit);
		}
	}
	dumpTime += loopCount;
	offset += loopCount * WARP_SIZE;

	if(forceExpand || (dumpTime >= (256 - loopCount))) {
		ExpandCounters(lane, warpShared, threadTotals);
		dumpTime = 0;
	}
}

template<typename T>
DEVICE2 void KSmallestCount(const T* source_global, uint* hist_global, 
	const int2* range_global, uint shift, uint bits) {

	__shared__ volatile uint counts_shared[NUM_THREADS * 16];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile uint* warpShared = counts_shared + 16 * WARP_SIZE * warp;
	ClearCounters(warpShared + lane);

	// Each warp holds two unpacked totals.
	uint threadTotals[2] = { 0, 0 };
	int dumpTime = 0;

	uint end = ROUND_DOWN(range.y, WARP_SIZE * INNER_LOOP_COUNT);

	// Use loop unrolling for the parts that aren't at the end of the array.
	// This reduces the logic for bounds checking.
	while(range.x < end)
		KSmallestCountItem(source_global, lane, range.x, end, shift, bits, 
			INNER_LOOP_COUNT, warpShared, threadTotals, false, dumpTime, false);

	// Process the end of the array. For an expansion of the counters.
	KSmallestCountItem(source_global, lane, range.x, range.y, shift, bits, 
		1, warpShared, threadTotals, true, dumpTime, true);

	hist_global[64 * gid + 2 * lane] = threadTotals[0];
	hist_global[64 * gid + 2 * lane + 1] = threadTotals[1];
}


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountUint(const uint* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	KSmallestCount(source_global, hist_global, range_global, shift, bits);

}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountInt(const int* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	KSmallestCount(source_global, hist_global, range_global, shift, bits);
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void KSmallestCountFloat(const float* source_global, uint* hist_global,
	const int2* range_global, uint shift, uint bits) {

	KSmallestCount(source_global, hist_global, range_global, shift, bits);
}


////////////////////////////////////////////////////////////////////////////////
// HISTOGRAM PASS

DEVICE uint MapToScan(uint k, uint x1, uint count1, uint x2, uint count2) {
	// Find the bucket that k maps to. This is the last bucket that starts at
	// or before k and has at least one element.
	uint low = __ballot(count1 && (k >= x1));
	uint high = __ballot(count2 && (k >= x2));

	uint kLow = __clz(low);
	uint kHigh = __clz(high);
	uint bucket = high ? (63 - kHigh) : (31 - kLow);

	return bucket;
}

//DEVICE void KSmallestIntervalScan(const uint* counts_global,
//	uint* scanWarp_global, uint k1, uint k2, uint lane, uint warp, 
//	volatile uint* shared,

// Run a scan on only the counts in bucket b from a single warp. May be called
// 
DEVICE void KSmallestScanWarp(const uint* counts_global, uint* scanWarp_global,
	uint b, uint lane, uint numWarps, volatile uint* shared) {

	shared[lane] = 0;
	volatile uint* s = shared + lane + WARP_SIZE / 2;
	uint offset = 0;

	for(int i = 0; i < numWarps; i += WARP_SIZE) {
		uint source = i + lane;
		uint count = 0;
		if(source < numWarps)
			count = counts_global[64 * source + b];

		// Parallel scan and write back to global memory.
		uint x = count;		
		s[0] = x;

		#pragma unroll
		for(int j = 0; j < LOG_WARP_SIZE; ++j) {
			int offset = 1<< j;
			x += s[-offset];
			s[0] = x;
		}

		// Make a coalesced write to scanWarp_global.
		scanWarp_global[i + lane] = offset + x - count;
		offset += shared[47];
	}
}

DEVICE void KSmallestHist(const uint* counts_global, uint* scanTotal_global,
	int numWarps, uint k1, uint k2, bool interval, uint* bucket1Scan_global, 
	uint* bucket2Scan_global, uint* intervalScan_global) {

	const int NumThreads = 1024;
	const int NumWarps = NumThreads / WARP_SIZE;

	__shared__ volatile uint shared[4 * 1024];
	volatile uint* counts_shared = shared;
	volatile uint* scan_shared = shared + 2 * 1024;
	volatile uint* b1_shared = scan_shared + 64;
	volatile uint* b2_shared = b1_shared + 1;

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	////////////////////////////////////////////////////////////////////////////
	// Loop over all counters and accumulate counts for each radix digits. This
	// is the sequential part.
	// Each thread maintains two counters (low and high lanes of the count).
	uint totals[2] = { 0, 0 };

	int curWarp = warp;
	while(curWarp < numWarps) {
		uint x1 = counts_global[64 * curWarp + lane];
		uint x2 = counts_global[64 * curWarp + 32 + lane];

		totals[0] += x1;
		totals[1] += x2;

		curWarp += NumWarps;
	}
	counts_shared[64 * warp + lane] = totals[0];
	counts_shared[64 * warp + 32 + lane] = totals[1];
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Have the first 64 threads run through shared memory and get final counts.
	// This is the parallel part.

	if(tid < 64) {
		uint total = counts_shared[tid];

		#pragma unroll
		for(int i = 64 + lane; i < 2048; i += 64)
			total += counts_shared[i];

		scan_shared[tid] = total;

		// Write the totals to global memory.
		scanTotal_global[tid] = total;
	}
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Parallel scan the radix digit totals. This is needed to find which radix
	// digit to expect k1 and k2 in. Those are stored in b1_shared and
	// b2_shared.

	// Scan the totals.
	if(tid < WARP_SIZE) {
		uint sum1 = scan_shared[tid];
		uint sum2 = scan_shared[WARP_SIZE + tid];

		uint x1 = sum1;
		uint x2 = sum2;

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			
			if(tid >= offset) x1 += scan_shared[tid - offset];
			x2 += scan_shared[WARP_SIZE + tid - offset];
			
			scan_shared[tid] = x1;
			scan_shared[WARP_SIZE + tid] = x2;
		}
		x2 += scan_shared[tid];

		x1 -= sum1;
		x2 -= sum2;

		// Write the scan to global memory.
		scanTotal_global[64 + tid] = x1;
		scanTotal_global[64 + WARP_SIZE + tid] = x2;

		uint b1 = MapToScan(k1, x1, sum1, x2, sum2);
		if(!tid) {
			*b1_shared = b1;
			scanTotal_global[2 * 64 + 0] = b1;
		}

		if(interval) {
			uint b2 = MapToScan(k2, x1, sum1, x2, sum2);
			if(!tid) {
				*b2_shared = b2;
				scanTotal_global[2 * 64 + 1] = b2;
			}
		}
	}
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Generate the warp scan offsets. There are three cases for this:
	// 1) Both k1 and k2 map to the same bucket. Only generate a single warp
	// scan array.
	// 2) k1 and k2 map to adjacent buckets. Here we scan only buckets b1 and 
	// b2.
	// 3) k1 and k2 map to adjacent buckets. Here we scan buckets b1, b2, and
	// also the sum of counts inside the interval over all warps.

	uint b1 = *b1_shared;
	uint b2;
	if(interval) b2 = *b2_shared;

	if(!interval || (b1 == b2)) {
		if(tid < 32) 
			KSmallestScanWarp(counts_global, bucket1Scan_global, b1, lane,
				numWarps, shared);		
	} else if(b1 + 1 == b2) {
		if(tid < 64) {
			uint b = warp ? b2 : b1;
			volatile uint* s = warp ? (shared + 128) : shared;
			uint* scanGlobal = warp ? bucket1Scan_global : bucket2Scan_global;

			// Run the k-smallest scan on the first two warps in parallel.
			KSmallestScanWarp(counts_global, scanGlobal, b, lane, numWarps, s);
		}
	} else {



	}

}



extern "C" __global__ __launch_bounds__(1024, 1)
void KSmallestHistValue(const uint* counts_global, uint* scanTotal_global,
	uint* scanWarps_global, int numWarps, uint k) {


	KSmallestHist(counts_global, scanTotal_global, numWarps, k, 0, false,
		scanWarps_global, 0, 0);
}

extern "C" __global__ __launch_bounds__(1024, 1)
void KSmallestHistInterval(const uint* counts_global, uint* scanTotal_global,
	uint* scanWarps_global, int numWarps, uint k1, uint k2) {

	KSmallestHist(counts_global, scanTotal_global, numWarps, k1, k2, true,
		scanWarps_global, scanWarps_global + numWarps, 
		scanWarps_global + 2 * numWarps);
}



////////////////////////////////////////////////////////////////////////////////
// STREAM PASS
/*
template<typename T>
DEVICE2 void StreamValues(volatile T* values, uint lane, int& start, int& end, 
	T*& values_global, bool all) {

	if(all) {


	} else if(end > WARP_SIZE * INNER_LOOP) {
		int last = ~(WARP_SIZE - 1) & end;

		// 


	}
}

template<typename T>
DEVICE2 void StreamPairs(volatile T* values, volatile uint* indices, uint lane,
	int& start, int& end, T*& values_global, T*& indices_global, bool all) {


}




template<typename T>
DEVICE2 void KSmallestStream(const T* source_global, const int2* range_global, 
	const int* streamOffset_global, T* target_global, uint mask, uint digit) {

	__shared__ volatile T counts_shared[NUM_THREADS * 16];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint gid = NUM_WARPS * block + warp;

	int2 range = range_global[gid];

	volatile T* warpShared = counts_shared + 16 * WARP_SIZE * warp;

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
	int end = ROUND_DOWN(range.y, WARP_SIZE);
	while(range.x < end) {
		
		T x = source_global[range.x + lane];
		uint masked = mask & ConvertToUint(x);
		bool stream = masked == digit;
						
		uint warpStream = __ballot(stream);
		int streamDest = streamNext + __popc(warpMask & warpStream);
		streamNext += __popc(warpStream);

		if(stream) warpShared[streamDest] = x;

		StreamValues(warpShared, lane, streamStart, streamNext, target_global, 
			false);
		
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




*/







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
