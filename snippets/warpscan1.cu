////////////////////////////////////////////////////////////////////////////////
// Parallel scan demonstration source. Performs parallel scan with conditionals
// over a single warp.

#define WARP_SIZE 32

extern "C" __global__ void WarpScan1(const int* values, int* inclusive, 
	int* exclusive) {

	__shared__ volatile int scan[WARP_SIZE];
	int tid = threadIdx.x;

	// Read from global memory.
	int x = values[tid];
	scan[tid] = x;

	// Run each pass of the scan.
	int sum = x;
	// #pragma unroll
	// for(int offset = 1; offset < WARP_SIZE; offset *= 2) {
	// This code generates 
	//  * Advisory: Loop was not unrolled, cannot deduce loop trip count"
	// We want to use the above iterators, but nvcc totally sucks. It only
	// unrolls loops when the conditional is simply incremented.
	#pragma unroll
	for(int i = 0; i < 5; ++i) {
		// Counting from i = 0 to 5 and shifting 1 by that number of bits 
		// generates the desired offset sequence of (1, 2, 4, 8, 16).
		int offset = 1<< i;

		// Add tid - offset into sum, if this does not put us past the beginning
		// of the array. Write the sum back into scan array.
		if(tid >= offset) sum += scan[tid - offset];
		scan[tid] = sum;
	}

	// Write sum to inclusive and sum - x (the original source element) to
	// exclusive.
	inclusive[tid] = sum;
	exclusive[tid] = sum - x;
}

