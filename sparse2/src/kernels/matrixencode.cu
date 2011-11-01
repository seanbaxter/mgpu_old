

////////////////////////////////////////////////////////////////////////////////


DEVICE void RepopulateSharedRows(uint tid, uint numValues, int row, 
	int& cachedCount, int& globalOffset, int& available, int consumed, int end,
	const int* rowIndices_global, volatile int* rowIndices_shared) {

	// Load up to numValues from rowIndices_global. Always load a multiple of
	// 32 values for coalescing.

	// Move the remaining values after consumption to the front of the array.
	if(tid >= consumed) rowIndices_shared[tid - consumed] = row;
	available -= consumed;

	int remaining2 = ~(WARP_SIZE - 1) & available;
	int count = min(numValues - remaining2, end - globalOffset);

	if(tid < load)
		rowIndices_shared[available + tid] = 
			rowIndices_global[globalOffset + tid];
	__syncthreads();
	
	globalOffset += count;
	available += count;
}

struct Advance {
	int valuesConsumed;
	int numRows;
	int firstRow;
	int lastRow;
	int tidRow;
};

// Returns the number of 
DEVICE Advance ScanRows(uint tid, int available, const int* rowIndices) {

	__shared__ volatile int shared[WARP_SIZE];
	__shared__ volatile int lastTidRow_shared;

	if(tid < WARP_SIZE) shared[tid] = 0;
	
	int firstRow = rowIndices[0];
	int threadRow = rowIndices[tid];
	int nextRow = rowIndices[1 + tid];
	
	// find the number of values to consume
	
	// threadTest and subsequentTest are true if the corresponding thread is out
	// of range due to row value.
	int distance = firstRow + WARP_SIZE;
	int threadTest = (tid < available) && (threadRow < distance);
	int nextTest = (tid + 1 < available) && (nextRow < distance);
	
	// The last in-range thread writes to LAST_ROW_TID
	if(threadTest > nextTest) lastTidRow_shared = tid;
	__syncthreads();
	
	int lastTid = lastTidRow_shared;
	if(tid > lastTid) threadRow = lastTidRow;
	if(tid + 1 > lastTid) nextRow = lastTidRow;
	
	int rowDelta = threadRow - firstRow;

	if(tid == numValues - 1) nextRow = 0x7fffffff;
	
	// Set the availability bit
	if(threadRow < nextRow) shared[rowDelta] = 1;
	__syncthreads();

	Advance advance;
	advance.valuesConsumed = lastTid + 1;
	advance.numRows = 0;
	advance.firstRow = firstRow;
	advance.lastRow = lastTid;
	advance.tidRow = threadRow;
	if(tid < WARP_SIZE) advance.numRows = __popc(__ballot(shared[tid]));
	
	return advance;
}




void DeviceEncode

extern "C" __global__ void DeviceEnco