
////////////////////////////////////////////////////////////////////////////////
// Multiscan function parameterized over the number of warps in the block. Uses
// shared memory passed in from caller.

template<int NumWarps>
DEVICE2 uint2 Multiscan(uint tid, uint x, volatile uint* warpShared) {

	const int LogNumWarps = LOG_BASE_2(NumWarps);
		
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

	__shared__ volatile uint totals_shared[2 * NumWarps];
	if(WARP_SIZE - 1 == lane) {
		totals_shared[NumWarps + warp] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NumWarps) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		uint total = totals_shared[NumWarps + tid];
		totals_shared[tid] = 0;
		volatile uint* s = totals_shared + NumWarps + tid;

		uint totalsSum = total;

		#pragma unroll
		for(int i = 0; i < LogNumWarps; ++i) {
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
	uint total = totals_shared[2 * NumWarps - 1];
	return make_uint2(sum, total);
}


////////////////////////////////////////////////////////////////////////////////
// Multiscan that allocates its own shared memory. More convenient, but shared
// memory is limited, so only use when this doesn't result in an occupancy 
// decrease.

template<int NumWarps>
DEVICE2 uint2 Multiscan2(uint tid, uint x) {
	uint warp = tid / WARP_SIZE;
	const int WarpStride = WARP_SIZE + WARP_SIZE / 2;
	const int SharedSize = NumWarps * WarpStride;
	__shared__ volatile uint shared[SharedSize];
	volatile uint* warpShared = shared + warp * WarpStride;
	return Multiscan<NumWarps>(tid, x, warpShared);
}


////////////////////////////////////////////////////////////////////////////////
// Reduction function for upsweep pass. This performs addition for code 0 and
// max for code 1.

template<int NumWarps>
DEVICE2 int Reduce(uint tid, int x, int code) {

	const int LogNumWarps = LOG_BASE_2(NumWarps);

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	int init = code ? -1 : 0;

	const int ScanStride = WARP_SIZE + WARP_SIZE / 2 + 1;
	const int ScanSize = NumWarps * ScanStride;
	__shared__ volatile int reduction_shared[ScanSize];
	__shared__ volatile int totals_shared[2 * WARP_SIZE];

	volatile int* s = reduction_shared + ScanStride * warp + lane +
		WARP_SIZE / 2;
	s[-(WARP_SIZE / 2)] = init;
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
	if(tid < NumWarps) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		x = reduction_shared[ScanStride * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

		volatile int* s = totals_shared + NumWarps / 2 + tid;
		s[-(NumWarps / 2)] = init;
		s[0] = x;

		#pragma unroll
		for(int i = 0; i < LogNumWarps; ++i) {
			int offset = 1<< i;
			if(0 == code) x += s[-offset];
			else if(1 == code) x = max(x, s[-offset]);
			if(i < LogNumWarps - 1) s[0] = x;
		}
		totals_shared[tid] = x;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	return totals_shared[NumWarps - 1];
}


