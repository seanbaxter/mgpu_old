#include <device_functions.h>
#include <vector_functions.h>

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define NUM_THREADS 256
#define BLOCKS_PER_SM 6
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)

#define LOG_NUM_WARPS 3

#define FLT_MAX 1.0e37f

#define DEVICE extern "C" __device__ __forceinline__

typedef unsigned int uint;


////////////////////////////////////////////////////////////////////////////////
// Reduction function to find the largest element in a block.

DEVICE void MaxIndexReduce(float& x, int& index, int tid) {

	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	const int Size = 49 * NUM_WARPS;

	__shared__ volatile float sharedX[Size];
	__shared__ volatile int sharedIndex[Size];

	volatile float* threadX = sharedX + 49 * warp + 16 + lane;
	volatile int* threadIndex = sharedIndex + 49 * warp + 16 + lane;

	// Zero out the preceding slots so we don't have to use conditionals.
	threadX[-16] = -FLT_MAX;

	// Store x and index into shared memory.
	threadX[0] = x;
	threadIndex[0] = index;
	
	// Ripped almost line-for-line from the globalscan tutorial.
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		// Use same scan indices as you would when finding the prefix sum.
		int offset = 1<< i;
		float y = threadX[-offset];
		int yIndex = threadIndex[-offset];
		if(y > x) {
			x = y;
			index = yIndex;
		}
		threadX[0] = x;
		threadIndex[0] = index;
	}

	// Synchronize and prepare for a inter-warp reduction.
	__syncthreads();
	
	if(tid < NUM_WARPS) {
		// Get the warp totals for warp tid.
		int warpIndex = 49 * tid + 16 + 31;
		x = sharedX[warpIndex];
		index = sharedIndex[warpIndex];

		sharedX[tid] = x;
		sharedIndex[tid] = index;

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			if(tid >= offset) {
				float y = sharedX[tid - offset];
				int yIndex = sharedIndex[tid - offset];
				if(y > x) {
					x = y;
					index = yIndex;
				}
			}
			sharedX[tid] = x;
			sharedIndex[tid] = index;
		}
	}
	__syncthreads();

	// Pull the final values from the reduction array. Do not perform the 
	// downsweep phase
	x = sharedX[NUM_WARPS - 1];
	index = sharedIndex[NUM_WARPS - 1];
}


////////////////////////////////////////////////////////////////////////////////
// Upsweep pass to find the max element and its index in the original sequence.

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void FindMaxIndexLoop(const float* data_global, float* max_global, 
	int* index_global, const uint2* range_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	uint2 range = range_global[block];

	// Start with the minimum possible float and set the index to 0.
	float maxX = -FLT_MAX;
	int maxIndex = 0;

	// Loop through all the values in the range. This is a sequential scan. If
	// we were to use an addition operator we'd have a normal prefix sum. The 
	// fact that we're searching for coupled values (max X and its index) does
	// not change the logic of the reduction.
	for(int i = range.x + tid; i < range.y; i += NUM_THREADS) {
		float x = data_global[i];
		if(x > maxX) {
			maxX = x;
			maxIndex = i;
		}
	}

	// Synchronize and run a parallel scan to find the max element of the entire
	// block.
	__syncthreads();

	MaxIndexReduce(maxX, maxIndex, tid);

	if(!tid) {
		max_global[block] = maxX;
		index_global[block] = maxIndex;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Reduce phase for max index - only runs on one block and returns the max
// from all the blocks run in the upsweep phase.

extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void FindMaxIndexReduce(float* max_global, int* index_global, uint numBlocks) {
	
	uint tid = threadIdx.x;

	// Start with the minimum possible float and set the index to 0.
	float maxX = -FLT_MAX;
	int maxIndex = 0;

	if(tid < numBlocks) {
		maxX = max_global[tid];
		maxIndex = index_global[tid];
	}

	// Synchronize and run a parallel scan to find the max element of the entire
	// block.
	__syncthreads();

	MaxIndexReduce(maxX, maxIndex, tid);

	if(!tid) {
		max_global[0] = maxX;
		index_global[0] = maxIndex;
	}
}
