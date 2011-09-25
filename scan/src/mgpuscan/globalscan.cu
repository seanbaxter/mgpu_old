#define NUM_THREADS SCAN_NUM_THREADS
#define VALUES_PER_THREAD SCAN_VALUES_PER_THREAD
#define BLOCKS_PER_SM SCAN_BLOCKS_PER_SM


#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS LOG_BASE_2(NUM_WARPS)
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)


////////////////////////////////////////////////////////////////////////////////
// Multiscan utility function. Used in the first and third passes of the
// global scan function. Returns the inclusive scan of the arguments in .x and
// the sum of all arguments in .y.

// Each warp is passed a pointer to its own contiguous area of shared memory.
// There must be at least 48 slots of memory. They should also be aligned so
// that the difference between the start of consecutive warps differ by an 
// interval that is relatively prime to 32 (any odd number will do).



////////////////////////////////////////////////////////////////////////////////
// GlobalScanUpsweep adds up all the values in elements_global within the 
// range given by blockCount and writes to blockTotals_global[blockIdx.x].

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void GlobalScanUpsweep(const uint* valuesIn_global, const uint2* range_global,
	uint* blockTotals_global) {

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint2 range = range_global[block];

	// Loop through all elements in the interval, adding up values.
	// There is no need to synchronize until we perform the multiscan.
	uint sum = 0;
	for(uint index = range.x + tid; index < range.y; index += 2 * NUM_THREADS)
		sum += valuesIn_global[index] + valuesIn_global[index + NUM_THREADS];

	// A full multiscan is unnecessary here - we really only need the total.
	// But this is easy and won't slow us down since this kernel is already
	// bandwidth limited.
	uint total = Multiscan2<NUM_WARPS>(tid, sum).y;

	if(!tid)
		blockTotals_global[block] = total;
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanReduction performs an exclusive scan on the elements in 
// blockTotals_global and writes back in-place.

extern "C" __global__ __launch_bounds__(REDUCTION_NUM_THREADS, 1)
void GlobalScanReduction(uint* blockTotals_global, 
	uint numBlocks) {

	uint tid = threadIdx.x;
	uint x = 0; 
	if(tid < numBlocks) x = blockTotals_global[tid];

	// Subtract the value from the inclusive scan for the exclusive scan.
	uint2 scan = Multiscan2<REDUCTION_NUM_THREADS / WARP_SIZE>(tid, x);
	if(tid < numBlocks) blockTotals_global[tid] = scan.x - x;

	// Have the first thread in the block set the scan total.
	if(!tid) blockTotals_global[numBlocks] = scan.y;
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanDownsweep runs an exclusive scan on the same interval of data as in
// pass 1, and adds blockScan_global[blockIdx.x] to each of them, writing back
// out in-place.

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void GlobalScanDownsweep(const uint* valuesIn_global, uint* valuesOut_global,
	const uint* blockScan_global, const int2* range_global, int count, 
	int inclusive) {

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint index = VALUES_PER_WARP * warp + lane;

	uint blockScan = blockScan_global[block];
	int2 range = range_global[block];

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	volatile uint* warpShared = shared + 
		warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* threadShared = warpShared + lane;

	// Transpose values into thread order.
	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;

	while(range.x < range.y) {

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint source = range.x + index + i * WARP_SIZE;
			uint x = valuesIn_global[source];

			threadShared[i * (WARP_SIZE + 1)] = x;
		}

		// Transpose into thread order by reading from transposeValues.
		// Compute the exclusive or inclusive scan of the thread values and 
		// their sum.
		uint scan[VALUES_PER_THREAD];
		uint sum = 0;
	
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = warpShared[offset + i];
			scan[i] = sum;
			if(inclusive) scan[i] += x;
			sum += x;
		}


		// Multiscan for each thread's scan offset within the block. Subtract
		// sum to make it an exclusive scan.
		uint2 localScan = Multiscan2<NUM_WARPS>(tid, sum);
		uint scanOffset = localScan.x + blockScan - sum;

		// Add the scan offset to each exclusive scan and put the values back
		// into the shared memory they came out of.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = scan[i] + scanOffset;
			warpShared[offset + i] = x;
		}

		// Store the scan back to global memory.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = threadShared[i * (WARP_SIZE + 1)];
			uint target = range.x + index + i * WARP_SIZE;
			valuesOut_global[target] = x;
		}

		// Grab the last element of totals_shared, which was set in Multiscan.
		// This is the total for all the values encountered in this pass.
		blockScan += localScan.y;

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
