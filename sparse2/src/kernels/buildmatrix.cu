#define WARP_SIZE 32
#define LOG_WA RP_SIZE 5

const int LAST_ROW_TID = 1;

#define DEVICE extern "C" __device__ __forceinline__ 
#define DEVICE2 __device__ __forceinline__ 

typedef unsigned int uint;

// Flags to begin a segmented scan and to commit the scan to shared mem.
const uint PARTIAL_STORE_BIT = 1<< 24;


// ONE BIT FOR EACH VALUE:
// PARTIAL_STORE_BIT = 1<< 24: Last occurence of a value of this row in the 
//		thread. When this is encountered, store the partial dot product to 
//		shared memory and zero the sum.

// FOUR OFFSETS TO COMPUTE FOR EACH WARP:

// 1) scan offset: This is the total number of PARTIAL_STORE_BIT flags 
//		encountered before the current thread. On encountering a 
//		PARTIAL_STORE_BIT, a thread stores to shared memory at the scan offset,
//		increments the scan offset, and zeroes the dot product running sum.

// 2) segscan distance x,
// 3) segscan distance y: The distance from tid or 32 + tid to the start of
//		the current segment. A segment is the contiguous set of partial dot
//		product entries. When a row's data spans multiple threads, each thread
//		will store the partial dot product for that row to shared memory. This 
//		flag allows the 64-wide parallel segmented scan to be executed
//		efficiently.

// 4) 	

// Each thread initializes pointers dynamically.
struct ThreadContext {
	// Reserve (WARP_SIZE + numValues) for the three data arrays. This allows us
	// to absolutely minimize global loads. Reserve 64 values for reduction 
	// array. By templating over NUM_VALUES, all of these pointers should be
	// treated as constant offsets by the compiler. They are for convenience
	// and won't eat into the register counts.
	uint* rowIndices;
	uint* colIndices;
	T* values;
	volatile uint* rowAvailability;			// WARP_SIZE
	volatile uint* lastThreadReduction;		// 2 * WARP_SIZE
	uint* lastTidRow;						// 1

	uint lane;
	uint warp;

	uint sharedScatter;
	uint sharedGather;

	// tid handles the value associated with (evalThread, evalValue) in the 
	// evaluation kernel.
	uint evalThread;
	uint evalValue;

	// Number of values sitting in the buffer.
	uint available;

	// Codes the thread inherits from its position within the block.
	uint threadCode;
};

DEVICE void ProcessRowIndices(uint tid, ThreadContext context) {


	__shared__ volatile int lastTid_shared;
	__shared__ volatile int rowStart_shared[WARP_SIZE];
	__shared__ volatile int firstRowInThread_shared[WARP_SIZE];

	if(tid < WARP_SIZE) {
		rowStart_shared[tid] = -1;



	}


	// NOTE: use an adjusted index for conflict-free access.
	int firstRow = context.rowIndices[0];
	int threadRow = context.rowIndices[tid];
	int subsequentRow = context.rowIndices[tid + 1];

	// endRow is one past the last row that can possibly be encountered in this
	// block.
	int endRow = firstRow + WARP_SIZE;

	// These values are true if the thread's value is in the block.
	int threadTest = (tid < context.available) && (threadRow < endRow);
	int subsequentTest = (tid + 1 < available) && (subsequentRow < endRow);

	// The last in-range thread writes to lastTid_shared.
	if(threadTest != subsequentTest) lastTid_shared = tid;
	__syncthreads();

	// Retrieve the tid of the last element in the block and its row index.
	uint lastTid = lastTid_shared;
	int lastRow = context.rowIndices[lastTid];
	if(tid > lastTid) threadRow = lastRow;
	if(tid + 1 > lastTid) subsequentRow = lastRow;
	if(tid == context.numValues - 1) subsequentRow = 0x7fffffff;
	
	int precedingRow = context.rowIndices[max(tid,  1) - 1];
	if(tid > lastTid) precedingRow = lastRow;
	if(!tid) precedingRow = -1;

	// If this is the first value of this row in the block, store to
	// rowStart_shared.
	if(precedingRow != threadRow) rowStart_shared[rowDelta] = evalThread;



	





	


}





	int rowDelta = threadRow - firstRow;

	int threadFlags = 0;
	if(precedingRow < threadRow)
		threadFlags |= LastThreadRow;
	reductionCode |= 1<< 25;

	// Prepare the segmented scan codes for this value.
	uint threadFlags = context.threadCode;
	
	// If this value is in a different row from the preceding one, it STARTS a
	// segment.
	if(precedingRow < threadRow)
		threadFlags |= FirstThreadRow;

	// If this value is in a different row from the subsequent one, it ENDS a
	// segment.
	if(threadRow < subsequentRow) {
		threadFlags |= LastThreadRow;

		// Mark that a particular value has been encountered.
		// NOTE: Is this necessary?
		context.rowAvailability[rowDelta] = 1;
	}

	if(LastThreadRow & threadFlags) {
		uint code = rowDelta | (context.evalThread<< 20) | (1<< 19);
		context.lastThreadReduction[context.evalThread + rowDelta] = code;
	}

	__syncthreads();
	return lastTid;
}


/*




template<int NUM_VALUES> 
ThreadContext<NUM_VALUES> BuildContext(uint tid, uint* shared) {

	// Include spacing in the three buffers to eliminate bank conflicts during
	// transpose.
	const int count = NUM_VALUES + (NUM_VALUES / WARP_SIZE) + WARP_SIZE;

	ThreadContext c;
	c.rowIndices = shared;
	c.colIndices = c.rowIndices + Count;
	c.values = (T*)(c.colIndices + Count);
	c.rowAvailability = (volatile uint*)(c.values + Count);
	c.lastThreadReduction = c.rowAvailability + WARP_SIZE;
	c.lastTidRow = c.lastThreadReduction + 2 * WARP_SIZE;

	c.lane = (WARP_SIZE - 1) & tid;
	c.warp = tid / WARP_SIZE;

	// Scatter/gather through shared memory to perform a conflict-free transpose
	// to put the data in thread order from strided order. This simplifies the
	// evaluation kernel.
	uint vt = NUM_VALUES / WARP_SIZE;
	uint sharedOffset = vt * c.lane;
	sharedOffset += sharedOffset / WARP_SIZE;
	c.sharedScatter = lane + warp * (WARP_SIZE + 1);
	c.sharedGather = sharedOffset + warp;

	// This division is a *slow* operation on GPU because there is no integer
	// division. However we only have to do this once per thread at the start
	// of the thread block. This value gets reused many times.
	c.evalThread = tid / vt;
	c.evalValue = tid - vt * c.evalThread;

	c.available = 0;
	c.threadCode = 0;
	if(0 == c.evalValue) c.threadCode |= FirstThreadRow;
	if(vt - 1 == c.evalValue) c.threadCode |= LastThreadRow;

	return c;
}


////////////////////////////////////////////////////////////////////////////////
// WriteThreadFlags

// Returns the last tid of the warp. 
DEVICE uint WriteThreadFlags(uint tid, ThreadContext context) {
	// These three row indices are enough to determine the head flags and if the
	// thread's value actually belongs in this block.
	int firstRow = context.rowIndices[0];
	int threadRow = context.rowIndices[tid];
	int subsequentRow = context.rowIndices[tid + 1];

	// endRow is one past the last row that can possibly be encountered in this
	// block.
	int endRow = firstRow + WARP_SIZE;

	// These values are true if the thread's value is in the block.
	int threadTest = (tid < context.available) && (threadRow < endRow);
	int subsequentTest = (tid + 1 < available) && (subsequentRow < endRow);

	// The last in-range thread writes to reduction[LAST_ROW_TID].
	if(threadTest != subsequentTest) *context.lastTidRow = tid;
	__syncthreads();

	// Retrieve the tid of the last element in the block and its row index.
	uint lastTid = *context.lastTidRow;
	int lastRow = context.rowIndices[lastTid];
	if(tid > lastTid) threadRow = lastRow;
	if(tid + 1 > lastTid) subsequentRow = lastRow;
	if(tid == context.numValues - 1) subsequentRow = 0x7fffffff;
	
	int precedingRow = context.rowIndices[max(tid,  1) - 1];
	if(tid > lastTid) precedingRow = lastRow;
	if(!tid) precedingRow = -1;

	int rowDelta = threadRow - firstRow;

	int threadFlags = 0;
	if(precedingRow < threadRow)
		threadFlags |= LastThreadRow;
	reductionCode |= 1<< 25;

	// Prepare the segmented scan codes for this value.
	uint threadFlags = context.threadCode;
	
	// If this value is in a different row from the preceding one, it STARTS a
	// segment.
	if(precedingRow < threadRow)
		threadFlags |= FirstThreadRow;

	// If this value is in a different row from the subsequent one, it ENDS a
	// segment.
	if(threadRow < subsequentRow) {
		threadFlags |= LastThreadRow;

		// Mark that a particular value has been encountered.
		// NOTE: Is this necessary?
		context.rowAvailability[rowDelta] = 1;
	}

	if(LastThreadRow & threadFlags) {
		uint code = rowDelta | (context.evalThread<< 20) | (1<< 19);
		context.lastThreadReduction[context.evalThread + rowDelta] = code;
	}

	__syncthreads();
	return lastTid;
}


////////////////////////////////////////////////////////////////////////////////

template<int NUM_VALUES>
DEVICE2 ComputeSegScan(uint tid, ThreadContext context) {
	

}



*/

