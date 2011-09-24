
#define NUM_THREADS 1024
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 5

#define BLOCKS_PER_SM 1

#define VALUES_PER_THREAD 4
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

DEVICE uint2 Multiscan(uint tid, uint x, volatile uint* warpShared) {

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	volatile uint* s = warpShared + lane + WARP_SIZE / 2;
	warpShared[lane] = 0;
	s[0] = x;

	// Run inclusive scan on each warp's data.
	uint sum = x;	
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		sum += s[-offset];
		if(i < LOG_WARP_SIZE - 1) s[0] = sum;
	}

	__shared__ volatile uint totals_shared[2 * NUM_WARPS];
	if(WARP_SIZE - 1 == lane) {
		totals_shared[NUM_WARPS + warp] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		uint total = totals_shared[NUM_WARPS + tid];
		totals_shared[tid] = 0;
		volatile uint* s = totals_shared + NUM_WARPS + tid;

		uint totalsSum = total;

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			totalsSum += s[-offset];
			s[0] = totalsSum;	
		}

		// Subtract total from totalsSum for an exclusive scan.
		totals_shared[tid] = totalsSum - total;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	// Add the block scan to the inclusive sum for the block.
	sum += totals_shared[warp];
	uint total = totals_shared[2 * NUM_WARPS - 1];
	return make_uint2(sum, total);
}

DEVICE uint2 Multiscan2(uint tid, uint x) {
	uint warp = tid / WARP_SIZE;
	const int WarpStride = WARP_SIZE + WARP_SIZE / 2;
	const int SharedSize = NUM_WARPS * WarpStride;
	__shared__ volatile uint shared[SharedSize];
	volatile uint* warpShared = shared + warp * WarpStride;
	return Multiscan(tid, x, warpShared);
}


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
	uint total = Multiscan2(tid, sum).y;

	if(!tid)
		blockTotals_global[block] = total;
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanReduction performs an exclusive scan on the elements in 
// blockTotals_global and writes back in-place.

extern "C" __global__ void GlobalScanReduction(uint* blockTotals_global, 
	uint numBlocks) {

	uint tid = threadIdx.x;
	uint x = 0; 
	if(tid < numBlocks) x = blockTotals_global[tid];

	// Subtract the value from the inclusive scan for the exclusive scan.
	uint2 scan = Multiscan2(tid, x);
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


	// Allocate 33 slots of shared memory per warp of data read. This allows
	// use to perform a conflict-free transpose from strided order to thread
	// order.
	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];

	// warpShared points to the start of the warp's data.
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
		uint2 localScan = Multiscan(tid, sum, warpShared);
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
