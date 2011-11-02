#include "common.cu"

////////////////////////////////////////////////////////////////////////////////

DEVICE void MoveToFront(uint tid, uint consumed, int& available, int numValues,
	volatile int* rowIndices_shared) {

	if(consumed < available) {
		// Move all the values left by consumed slots.
		int tidRow1 = rowIndices_shared[tid];
		int tidRow2;
		if(tid < WARP_SIZE) 
			tidRow2 = rowIndices_shared[numValues + tid];
		__syncthreads();

		if(tid < WARP_SIZE)
			rowIndices_shared[numValues + tid - consumed] = tidRow2;
		if(tid >= consumed)
			rowIndices_shared[tid - consumed] = tidRow1;
	}
	available -= consumed;
}

DEVICE void RepopulateSharedRows(uint tid, int numValues, int& available, 
	int& remaining, const int*& rowIndices_global, 
	volatile int* rowIndices_shared) {

	// Load up to numValues from rowIndices_global. Always load a multiple of
	// 32 values for coalescing.
	int remaining2 = ~(WARP_SIZE - 1) & available;
	int count = min(numValues - remaining2, remaining);

	if(tid < count)
		rowIndices_shared[available + tid] = rowIndices_global[tid];
	__syncthreads();
	
	rowIndices_global += count;
	available += count;
}

struct Advance {
	int valuesConsumed;
	int numRows;
	int firstRow;
	int lastTidRow;
};

// Returns the number of 
DEVICE Advance ScanRows(uint tid, int available, const int* rowIndices) {

	__shared__ volatile int shared[WARP_SIZE];
	__shared__ volatile int lastTid_shared;

	if(tid < WARP_SIZE) shared[tid] = 0;
	
	int firstRow = rowIndices[0];
	int tidRow = rowIndices[tid];
	int nextRow = rowIndices[1 + tid];
	
	// find the number of values to consume
	
	// threadTest and subsequentTest are true if the corresponding thread is out
	// of range due to row value.
	int distance = firstRow + WARP_SIZE;
	int threadTest = (tid < available) && (tidRow < distance);
	int nextTest = (tid + 1 < available) && (nextRow < distance);
	
	// The last in-range thread writes to LAST_ROW_TID
	if(threadTest > nextTest) lastTid_shared = tid;
	__syncthreads();
	
	int lastTid = lastTid_shared;
	int lastTidRow = rowIndices[lastTid];
	if(tid > lastTid) tidRow = lastTidRow;
	if(tid + 1 > lastTid) nextRow = lastTidRow;
	
	int rowDelta = tidRow - firstRow;
	
	// Set the availability bit
	if(tidRow < nextRow) shared[rowDelta] = 1;
	__syncthreads();

	Advance advance;
	advance.valuesConsumed = lastTid + 1;
	advance.numRows = 0;
	advance.firstRow = firstRow;
	advance.lastTidRow = lastTidRow;
	if(tid < WARP_SIZE) advance.numRows = __popc(__ballot(shared[tid]));
	
	return advance;
}


template<int VT>
DEVICE2 void MatrixCount(const int2* rangePairs_global,
	const int* rowIndices_global, int4* groupInfoOut_global) {

	const int Count = WARP_SIZE * VT;

	__shared__ int rowIndices_shared[Count];

	uint tid = threadIdx.x;
	uint block = blockIdx.x;

	int2 rangePair = rangePairs_global[block];
	int remaining = rangePair.y - rangePair.x;
	int available = 0;

	// group info terms.
	int groupCount = 0;
	int uniqueCount = 0;
	int lastGroupRow = -1;

	while(remaining) {
		RepopulateSharedRows(tid, Count, available, remaining,
			rowIndices_global, rowIndices_shared);
		
		Advance advance = ScanRows(tid, available, rowIndices_shared);
		if(!tid) {
			++groupCount;
			uniqueCount += advance.numRows;
			lastGroupRow = advance.lastTidRow;
		}

		MoveToFront(tid, advance.valuesConsumed, available, Count, 
			rowIndices_shared);
	}

	if(!tid) 
		groupInfoOut_global[block] = 
			make_int4(groupCount, uniqueCount, lastGroupRow, 0);
}

#define GEN_COUNT(count)												\
extern "C" __global__ void MatrixCount_##count(							\
	const int2* rangePairs_global, const int* rowIndices_global,		\
	int4* groupInfoOut_global) {										\
																		\
	MatrixCount<count>(rangePairs_global, rowIndices_global,			\
		groupInfoOut_global);											\
}

GEN_COUNT(4)
GEN_COUNT(6)
GEN_COUNT(8)
GEN_COUNT(10)
GEN_COUNT(12)
GEN_COUNT(16)
GEN_COUNT(20)
