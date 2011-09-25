#define NUM_THREADS PACKED_NUM_THREADS
#define BLOCKS_PER_SM PACKED_BLOCKS_PER_SM
#define VALUES_PER_THREAD PACKED_VALUES_PER_THREAD


#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS LOG_BASE_2(NUM_WARPS)
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)


////////////////////////////////////////////////////////////////////////////////
// UPSWEEP PASS. Find the sum of all values in the last segment in each block.
// When the first head flag in the block is encountered, write out the sum to 
// that point and return. We only need to reduce the last segment to feed sums
// up to the reduction pass.

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepPacked(const uint* packedIn_global, uint* blockLast_global,
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
			packed[i] = packedIn_global[current + tid + i * NUM_THREADS];


		// Find the index of the latest value loaded with a head flag set.
		int lastHeadFlagPos = -1;

		#pragma unroll
		for(int i = 0; i < UpsweepValues; ++i) {
			uint flag = 0x80000000 & packed[i];
			if(flag) lastHeadFlagPos = i;
		}
		if(-1 != lastHeadFlagPos)
			lastHeadFlagPos = tid + lastHeadFlagPos * NUM_THREADS;

		segmentStart = Reduce<NUM_WARPS>(tid, lastHeadFlagPos, 1);

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
	uint total = (uint)Reduce<NUM_WARPS>(tid, (int)threadSum, 0);

	if(0 == tid) {
		blockLast_global[block] = total;
		int headFlag = -1 != segmentStart;
		if(-1 != segmentStart) segmentStart += current;
		headFlagPos_global[block] = headFlag;
	}
}


////////////////////////////////////////////////////////////////////////////////
// DOWNSWEEP PASS.

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanDownsweepPacked(const uint* packedIn_global, uint* valuesOut_global,
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

		////////////////////////////////////////////////////////////////////////
		// Load and transpose values.

			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint source = range.x + index + i * WARP_SIZE;
				uint packed = packedIn_global[source];
				threadShared[i * (WARP_SIZE + 1)] = packed;
			}

		// Transpose into thread order and separate values from head flags.

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint packed = warpShared[offset + i];
			x[i] = 0x7fffffff & packed;
			flags[i] = 0x80000000 & packed;
		}


		////////////////////////////////////////////////////////////////////////
		// Run downsweep function on values and head flags.

		SegScanDownsweep<NUM_WARPS, VALUES_PER_THREAD>(tid, lane, warp, x, 
			flags, warpShared, threadShared, inclusive, &blockOffset_shared);

		////////////////////////////////////////////////////////////////////////
		// Transpose and store scanned values.

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i)
			warpShared[offset + i] = x[i];

			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint target = range.x + index + i * WARP_SIZE;
				valuesOut_global[target] = threadShared[i * (WARP_SIZE + 1)];
			}

		range.x += NUM_VALUES;
	}
}


#undef NUM_THREADS
#undef NUM_WARPS
#undef LOG_NUM_WARPS
#undef BLOCKS_PER_SM
#undef VALUES_PER_THREAD
#undef VALUES_PER_WARP
#undef NUM_VALUES

