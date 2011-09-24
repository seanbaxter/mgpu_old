#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 3

#define BLOCKS_PER_SM 2

#define VALUES_PER_THREAD 16
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)

#include "segscancommon.cu"


////////////////////////////////////////////////////////////////////////////////
// UPSWEEP PASS. Find the sum of all values in the last segment in each block.
// When the first head flag in the block is encountered, write out the sum to 
// that point and return. We only need to reduce the last segment to feed sums
// up to the reduction pass.


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepFlag(const uint* valuesIn_global, uint* blockLast_global,
	int* headFlagPos_global, const int2* rangePairs_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	int2 range = rangePairs_global[block];

	const int UpsweepValues = 8;
	const int NumValues = UpsweepValues * NUM_THREADS;
	// Start at the last tile (NUM_VALUES before the end iterator). Because
	// upsweep isn't executed for the last block, we don't have to worry about
	// the ending edge case.
	int current = range.y - NumValues;

	uint threadSum = 0;
	int segmentStart = -1;


	while(current >= range.x) {

		uint packed[UpsweepValues];
	
		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) 
			packed[i] = valuesIn_global[current + tid + i * NUM_THREADS];


		// Find the index of the latest value loaded with a head flag set.
		int lastHeadFlagPos = -1;

		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) {
			uint flag = 0x80000000 & packed[i];
			if(flag) lastHeadFlagPos = i;
		}
		if(-1 != lastHeadFlagPos)
			lastHeadFlagPos = tid + lastHeadFlagPos * NUM_THREADS;

		segmentStart = Reduce(tid, lastHeadFlagPos, 1, -1);

		// Make a second pass and sum all the values that appear at or after
		// segmentStart.

		// Add if tid + i * NUM_THREADS >= segmentStart.
		// Subtract tid from both sides to simplify expression.
		int cmp = segmentStart - tid;
		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) {
			uint value = 0x7fffffff & packed[i];
			if(i * NUM_THREADS >= cmp)
				threadSum += value;
		}
		if(-1 != segmentStart) break;

		current -= NumValues;
	}

	// We've either hit the head flag or run out of values. Do a horizontal sum
	// of the thread values and store to global memory.
	uint total = (uint)Reduce(tid, (int)threadSum, 0, 0);

	if(0 == tid) {
		blockLast_global[block] = total;
		int headFlag = -1 != segmentStart;
		if(-1 != segmentStart) segmentStart += current;
		headFlagPos_global[block] = headFlag;
	}
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepKeys(const uint* valuesIn_global, const uint* keysIn_global,
	uint* blockLast_global, const int2* rangePairs_global) {


}


////////////////////////////////////////////////////////////////////////////////
// REDUCTION PASS. 

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void SegScanReduction(const uint* headFlags_global, uint* blockLast_global,
	uint numBlocks) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	// Load the head flag and last segment counts for each thread. These map
	// to blocks in the upsweep/downsweep passes.
	uint flag = 0;
	uint x = 0;
	if(tid < numBlocks) {
		flag = headFlags_global[tid];
		x = blockLast_global[tid];
	}

	// Get the start flags for each thread in the warp.
	uint flags = __ballot(flag);

	// Mask out the bits at or above the current lane.
	uint mask = bfi(0, 0xffffffff, 0, lane);
	uint flagsMasked = flags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	int preceding = 31 - __clz(flagsMasked);
	uint distance = lane - preceding;

	__shared__ volatile uint shared[NUM_WARPS * (WARP_SIZE + 1)];
	__shared__ volatile uint blockShared[2 * NUM_WARPS];
	volatile uint* warpShared = shared + warp * (WARP_SIZE + 1) + 1;
	volatile uint* threadShared = warpShared + lane;

	// Run an inclusive scan for each warp. This does not require any special 
	// treatment of segment edges, as we have only one value per thread.
	threadShared[-1] = 0;
	threadShared[0] = x;
	uint sum = x;
	uint first = warpShared[preceding];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		if(distance > offset)
			sum += threadShared[-offset];
		threadShared[0] = sum; 
	}
	sum += first;

	uint last = flag ? x : sum;

	// sum now holds the inclusive scan for the part of the segment within the
	// warp. Run a multiscan by having each warp store its flags value to
	// shared memory.
	if(WARP_SIZE - 1 == lane) {
		blockShared[warp] = flags;
		blockShared[NUM_WARPS + warp] = last;
	}

	__syncthreads();
	if(tid < NUM_WARPS) {

		// Load the inclusive sums for the last value in each warp and the head
		// flags for each warp.
		uint flag = blockShared[tid];
		uint x = blockShared[NUM_WARPS + tid];
		uint flags = __ballot(flag) & mask;

		int preceding = 31 - __clz(flags);
		uint distance = tid - preceding;

		volatile uint* s = blockShared + NUM_WARPS + tid;
		s[-NUM_WARPS] = 0;

		uint sum = x;
		uint first = blockShared[NUM_WARPS + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(distance > offset) sum += s[-offset];
			s[0] = sum;
		}

		// Add preceding and subtract x to get an exclusive sum.
		sum += first - x;

		blockShared[tid] = sum;
	}

	__syncthreads();

	uint blockScan = blockShared[warp];

	// Add blockScan if the warp doesn't hasn't encountered a head flag yet.
	if(!flagsMasked) sum += blockScan;
	sum -= x;

	if(tid < numBlocks)
		blockLast_global[tid] = sum;
}



extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepFlag(const uint* valuesIn_global, uint* valuesOut_global,
	const uint* start_global, const int2* rangePairs_global, int count,
	int inclusive) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint index = VALUES_PER_WARP * warp + lane;

	int2 range = rangePairs_global[block];

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];
	__shared__ volatile uint blockOffset_shared;

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Transpose values into thread order.
	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;

	int lastOffset = ~(NUM_VALUES - 1) & count;

	if(!tid) blockOffset_shared = start_global[block];

	while(range.x < range.y) {
		// Load values into packed.
		uint x[VALUES_PER_THREAD];
		uint flags[VALUES_PER_THREAD];

			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint source = range.x + index + i * WARP_SIZE;
				uint x = valuesIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = x;
			}

		// Transpose into thread order 
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint packed = warpShared[offset + i];
			x[i] = 0x7fffffff & packed;
			flags[i] = 0x80000000 & packed;
		}

		SegScanDownsweep(tid, lane, warp, x, flags, warpShared, 
			threadShared, inclusive, &blockOffset_shared);

		// Transpose 
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			warpShared[offset + i] = x[i];

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint target = range.x + index + i * WARP_SIZE;
			valuesOut_global[target] = threadShared[i * (WARP_SIZE + 1)];
		}




		// Load values
	/*	if(range.x >= lastOffset) {
			// Use conditional loads.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint x = 0;
				uint source = range.x + index + i * WARP_SIZE;
				if(source < count) x = valuesIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = x;
			}
		} else {*/
		//}


		// Store values
		/*if(range.x >= lastOffset) {
			// Use conditional loads.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint x = 0;
				uint source = range.x + index + i * WARP_SIZE;
				if(source < count) x = valuesIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = x;
			}
		} else {*/

	//	}

		range.x += NUM_VALUES;
	}
}




extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepKeys(const uint* valuesIn_global, uint* valuesOut_global,
	const uint* start_global, const int2* rangePairs_global, int count,
	uint init, bool inclusive) {

}

#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES
