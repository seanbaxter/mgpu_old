////////////////////////////////////////////////////////////////////////////////
// Parallel scan demonstration source. Performs parallel scan without 
// conditionals over a single warp.

#define WARP_SIZE 32

extern "C" __global__ void WarpScan2(const int* values, int* inclusive, 
	int* exclusive) {

	// Reserve a half warp of extra space.
	__shared__ volatile int scan[WARP_SIZE + WARP_SIZE / 2];
	int tid = threadIdx.x;

	// Zero out the values at the start of the scan array and make a pointer
	// to scan + WARP_SIZE / 2. Now s is addressable and zero for the first
	// sixteen values before the start of the array. This eliminates comparisons
	// and predicated execution.
	scan[tid] = 0;
	volatile int* s = scan + WARP_SIZE / 2 + tid;

	// Read from global memory.
	int x = values[tid];
	s[0] = x;

	// Run each pass of the scan.
	int sum = x;
	#pragma unroll
	for(int i = 0; i < 5; ++i) {
		int offset = 1<< i;

		// Since offset is at most 16, -offset will not read past the beginning
		// of the scan array.
		int y = s[-offset];
		
		sum += y;
		s[0] = sum;
	}

	// Write sum to inclusive and sum - x (the original source element) to
	// exclusive.
	inclusive[tid] = sum;
	exclusive[tid] = sum - x;
}

