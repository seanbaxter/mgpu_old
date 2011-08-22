///////////////////////////////////////////////////////////////////////////////
// Multiscan demonstration source.

#define WARP_SIZE 32
#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 8
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)

#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

extern "C" __global__ void Multiscan(const int* values, int* inclusive, 
	int* exclusive) {

	// Reserve a half warp of extra space plus one per warp in the block.
	// This is exactly enough space to avoid comparisons in the multiscan
	// and to avoid bank conflicts.
	__shared__ volatile int scan[NUM_WARPS * SCAN_STRIDE];
	int tid = threadIdx.x;
	int warp = tid / WARP_SIZE;
	int lane = (WARP_SIZE - 1) & tid;

	volatile int* s = scan + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
	s[-16] = 0;
	
	// Read from global memory.
	int x = values[tid];
	s[0] = x;

	// Run inclusive scan on each warp's data.
	int sum = x;	
	#pragma unroll
	for(int i = 0; i < 5; ++i) {
		int offset = 1<< i;
		sum += s[-offset];
		s[0] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	__shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		int total = scan[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

		totals[tid] = 0;
		volatile int* s2 = totals + NUM_WARPS / 2 + tid;
		int totalsSum = total;
		s2[0] = total;

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			totalsSum += s2[-offset];
			s2[0] = totalsSum;	
		}

		// Subtract total from totalsSum for an exclusive scan.
		totals[tid] = totalsSum - total;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	// Add the block scan to the inclusive sum for the block.
	sum += totals[warp];

	// Write the inclusive and exclusive scans to global memory.
	inclusive[tid] = sum;
	exclusive[tid] = sum - x;
}

