
#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 3

#define BLOCKS_PER_SM 4

#define VALUES_PER_THREAD 8
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)

// Use a 33-slot stride for shared mem transpose.
#define WARP_STRIDE (WARP_SIZE + 1)


////////////////////////////////////////////////////////////////////////////////
// UPSWEEP PASS. Find the sum of all values in the last segment in each block.
// When the first head flag in the block is encountered, write out the sum to 
// that point and return. We only need to reduce the last segment to feed sums
// up to the reduction pass.

DEVICE int Reduce(uint tid, int x, int code, int init) {

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	const int ScanStride = WARP_SIZE + WARP_SIZE / 2 + 1;
	const int ScanSize = NUM_WARPS * ScanStride;
	__shared__ volatile int reduction_shared[ScanSize];
	__shared__ volatile int totals_shared[NUM_WARPS + NUM_WARPS / 2];

	volatile int* s = reduction_shared + ScanStride * warp + lane + 
		WARP_SIZE / 2;
	s[-16] = init;
	s[0] = x;

	// Run intra-warp max reduction.
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(0 == code) x += s[-offset];
		else if(1 == code) x = max(x, s[-offset]);
		s[0] = x;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		x = reduction_shared[ScanStride * tid + WARP_SIZE / 2 +
			WARP_SIZE - 1];

		totals_shared[tid] = init;
		volatile int* s2 = totals_shared + NUM_WARPS / 2 + tid;
		s2[0] = x;
		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(0 == code) x += s[-offset];
			else if(1 == code) x = max(x, s[-offset]);
			s2[0] = x;
		}

		if(NUM_WARPS - 1 == tid) totals_shared[0] = x;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	return totals_shared[0];
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepFlag(const uint* valuesIn_global, uint* valuesOut_global,
	int* headFlagPos_global, const int2* rangePairs_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	int2 range = rangePairs_global[block];
	
	// Round the end iterator down to a multiple of NUM_VALUES.
	int current = ~(NUM_VALUES - 1) & (range.y - 1);

	// range.x was set by the host to be aligned. We need only check the ranges
	// for the last brick in each block (that is, the first one processed).
	bool checkRange = (block == gridDim.x - 1);
	
	uint threadSum = 0;
	int segmentStart = -1;

	while(current >= range.x) {

		uint packed[VALUES_PER_THREAD];
		if(checkRange) {
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				int index = current + tid + i * NUM_THREADS;
				packed[i] = 0;
				if(index < range.y)
					packed[i] = valuesIn_global[index];
			}
		} else {
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) 
				packed[i] = valuesIn_global[current + tid + i * NUM_THREADS];
		}

		// Find the index of the latest value loaded with a head flag set.
		int lastHeadFlagPos = -1;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
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
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint value = 0x7fffffff & packed[i];
			if(i * NUM_THREADS >= cmp)
				threadSum += value;
		}
		if(-1 != segmentStart) break;

		current -= NUM_VALUES;
		checkRange = false;
	}

	// We've either hit the head flag or run out of values. Do a horizontal sum
	// of the thread values and store to global memory.
	uint total = (uint)Reduce(tid, (int)threadSum, 0, 0);

	if(0 == tid) {
		valuesOut_global[block] = total;
		if(-1 != segmentStart) segmentStart += current;
		headFlagPos_global[block] = segmentStart;
	}
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepKeys(const uint* valuesIn_global, const uint* keysIn_global,
	uint* valuesOut_global, const int2* rangePairs_global) {


}


////////////////////////////////////////////////////////////////////////////////
// REDUCTION PASS. 

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void SegScanReduction(const uint* headFlags_global, uint* sums_global,
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
		x = sums_global[tid];
	}

	// Get the start flags for each thread in the warp.
	uint flags = __ballot(flag);

	// Mask out the bits above the current lane.
	uint mask = bfi(0, 0xffffffff, 0, lane + 1);
	uint flagsMasked = flags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	uint distance = __clz(flagsMasked) + lane - 31;

	__shared__ volatile uint shared[NUM_WARPS * (WARP_SIZE + 1)];
	__shared__ volatile uint blockShared[NUM_WARPS];
	volatile uint* warpShared = shared + warp * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Run an inclusive scan for each warp. This does not require any special 
	// treatment of segment edges, as we have only one value per thread.
	threadShared[0] = x;
	uint sum = x;
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		int offset = 1<< i;
		if(distance >= offset)
			sum += threadShared[-offset];
		threadShared[0] = offset; 
	}

	// sum now holds the inclusive scan for the part of the segment within the
	// warp. Run a multiscan by having each warp store its flags value to
	// shared memory.
	if(!lane) blockShared[warp] = flags;

	__syncthreads();
	if(tid < NUM_WARPS) {
		// Load the inclusive sums for the last value in each warp and the head
		// flags for each warp.
		uint x = shared[tid * (WARP_SIZE + 1) + WARP_SIZE - 1];
		uint flag = blockShared[tid];
		uint flags = __ballot(flag) & mask;

		int preceding = 31 - __clz(flags);
		uint distance = tid - preceding;
		
		shared[0] = 0;
		volatile uint* s = shared + tid + 1;
		s[0] = x;
		uint sum = x;
		uint first = shared[1 + preceding];
		
		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(distance > offset) sum += s[-offset];
			s[0] = sum;
		}
		
		// Add preceding and subtract x to get an exclusive sum.
		sum += preceding - x;

		shared[tid] = sum;
	}
	__syncthreads();

	uint blockScan = shared[warp];

	// Add blockScan if the warp doesn't hasn't encountered a head flag yet.
	if(flagsMasked) sum += blockScan;
	sum -= x;

	sums_global[tid] = x;
}


////////////////////////////////////////////////////////////////////////////////
// INTER-WARP REDUCTION 
// Calculate the length of the last segment in the last lane in each warp.

DEVICE uint BlockScan(uint warp, uint lane, uint last, uint warpFlags, 
	uint mask, volatile uint* shared, volatile uint* threadShared) {

	__shared__ volatile uint blockShared[3 * NUM_WARPS];
	if(WARP_SIZE - 1 == lane) {
		blockShared[NUM_WARPS + warp] = last;
		blockShared[2 * NUM_WARPS + warp] = warpFlags;
	}
	__syncthreads();

	if(lane < NUM_WARPS) {
		// Pull out the sum and flags for each warp.
		volatile uint* s = blockShared + NUM_WARPS + lane;
		uint warpLast = blockShared[NUM_WARPS + lane];
		uint flag = blockShared[2 * NUM_WARPS + lane];
		blockShared[lane] = 0;

		uint blockFlags = __ballot(flag);

		// Mask out the bits at or above the current warp.
		blockFlags &= mask;

		// Find the distance from the current warp to the warp at the start of 
		// this segment.
		int preceding = 31 - __clz(blockFlags);
		uint distance = lane - preceding;

		// INTER-WARP reduction
		uint warpSum = warpLast;
		uint warpFirst = blockShared[NUM_WARPS + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			uint offset = 1<< i;
			if(distance > offset) warpSum += s[-offset];
			s[0] = warpSum;
		}
		// Subtract warpLast to make exclusive and add first to grab the
		// fragment sum of the preceding warp.
		warpSum += warpFirst - warpLast;

		// Store warpSum back into shared memory. This is added to all the
		// lane sums and those are added into all the threads in the first 
		// segment of each lane.
		blockShared[lane] = warpSum;
	}
	__syncthreads();

	return blockShared[warp];
}



////////////////////////////////////////////////////////////////////////////////
// DOWNSWEEP PASS. Add the 


extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepFlag(const uint* valuesIn_global, uint* valuesOut_global,
	const uint* start_global, int count, const int2* rangePairs_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint index = VALUES_PER_WARP * warp + lane;

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];
	__shared__ volatile uint blockShared[3 * NUM_WARPS];

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;
	
	int lastOffset = ~(NUM_VALUES - 1) & count;

	int2 range = rangePairs_global[block];

	uint last = 0;
	if(!tid) last = start_global[block];

	while(range.x < range.y) {
		uint packed[VALUES_PER_THREAD];

		// Transpose values into thread order.
		uint offset = VALUES_PER_THREAD * lane;
		offset += offset / WARP_SIZE;

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
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint source = range.x + index + i * WARP_SIZE;
				uint x = valuesIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = x;
			}
		//}

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			packed[i] = warpShared[offset + i];
	

		////////////////////////////////////////////////////////////////////////
		// INTRA-WARP UPSWEEP PASS
		// Run a sequential segmented scan for all values in the packed array. 
		// Find the sum of all values in the thread's last segment. Additionally
		// set index to tid if any segments begin in this thread.
		
		uint hasHeadFlag = 0;

		uint x[VALUES_PER_THREAD];
		uint flags[VALUES_PER_THREAD];

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			flags[i] = 0x80000000 & packed[i];
			x[i] = 0x7fffffff & packed[i];
			if(flags[i]) last = 0;
			hasHeadFlag |= flags[i];
			last += x[i];
		}


		////////////////////////////////////////////////////////////////////////
		// INTRA-WARP SEGMENT PASS
		// Run a ballot and clz to find the lane containing the start value for
		// the segment that begins this thread.

		uint warpFlags = __ballot(hasHeadFlag);

		// Mask out the bits at or above the current thread.
		uint mask = bfi(0, 0xffffffff, 0, lane);
		uint warpFlagsMask = warpFlags & mask;

		// Find the distance from the current thread to the thread at the start
		// of the segment.
		int preceding = 31 - __clz(warpFlagsMask);
		uint distance = lane - preceding;


		////////////////////////////////////////////////////////////////////////
		// REDUCTION PASS
		// Run a prefix sum scan over last to compute for each lane the sum of
		// all values in the segmented preceding the current lane, up to that
		// point. This is added back into the thread-local exclusive scan for 
		// the continued segment in each thread.
		
		volatile uint* shifted = threadShared + 1;
		shifted[-1] = 0;
		shifted[0] = last;
		uint sum = last;
		uint first = warpShared[1 + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			uint offset = 1<< i;
			if(distance > offset) sum += shifted[-offset];
			shifted[0] = sum;
		}
		// Subtract last to make exclusive and add first to grab the fragment
		// sum of the preceding thread.
		sum += first - last;

		// Call BlockScan for inter-warp scan on the reductions of the last
		// segment in each warp.
		uint lastSegLength = last;
		if(!hasHeadFlag) lastSegLength += sum;

		uint blockScan = BlockScan(warp, lane, lastSegLength, warpFlags, mask, 
			shared, threadShared);
		if(!warpFlagsMask) sum += blockScan;


		////////////////////////////////////////////////////////////////////////
		// INTRA-WARP PASS
		// Add sum to all the values in the continuing segment (that is, before
		// the first start flag) in this thread.

		last = sum;

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			if(flags[i]) last = 0;

			// NOTE: if inclusive, add x into last before storing to shared mem.

			warpShared[offset + i] = last;
			last += x[i];
		}

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
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint target = range.x + index + i * WARP_SIZE;
				valuesOut_global[target] = threadShared[i * (WARP_SIZE + 1)];
			}
	//	}

		range.x += NUM_VALUES;
	}
}


////////////////////////////////////////////////////////////////////////////////
