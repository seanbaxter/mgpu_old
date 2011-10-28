#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define DEVICE extern "C" __device__ __forceinline__ 
#define DEVICE2 __device__ __forceinline__ 

typedef unsigned int uint;

// retrieve numBits bits from x starting at bit
DEVICE uint bfe(uint x, uint bit, uint numBits) {
	uint ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


// insert the first numBits of y into x starting at bit
DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}



// Flags to begin a segmented scan and to commit the scan to shared mem.
const uint STORE_FLAG = 1<< 24;

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

	uint numValues;
	uint vt;

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

	
	// Help in converting strided index to thread index. If valuesPerThread is
	// odd, we can simply dereference into shared memory without any bank 
	// conflicts. If valuesPerThread is a power of 2, we use a stride of 33 
	// (add offset / WARP_SIZE). If valuesPerThread is even but not a power of
	// two, we'll have to eat a two-way serialization penalty, but can use the
	// native stride of 32.
	bool stridedIndex;
	int Index(int offset) const {
		if(stridedIndex) offset += offset / WARP_SIZE;
		return offset;
	}
};

DEVICE void ProcessRowIndices(uint tid, ThreadContext context) {
	__shared__ volatile int tempSpace_shared[4 * WARP_SIZE];
	volatile int* first_shared = tempSpace_shared;
	volatile int* last_shared = tempSpace_shared + 32;
	volatile int& lastTid_share = tempSpace_shared[64];

	// Clear the rowStartThread and rowEndThread arrays. Any index other than 
	// -1 indicates that the row in question is in the block.
	if(tid < 2 * WARP_SIZE)
		rowStartThread_shared[tid] = -1;
	__syncthreads();


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
	
	int precedingRow = context.rowIndices[tid - 1];
	if(tid > lastTid) precedingRow = lastRow;
	if(!tid) precedingRow = -1;

	int nextRow = context.rowIndices[tid + 1];
	if(tid + 1 >= lastTid) nextRow = -1;

	// If this is the first value of this row in the block, store to
	// rowStart_shared.
	if(precedingRow != threadRow) first_shared[rowDelta] = context.evalThread;
	if(threadRow != nextRow)
		last_shared[rowDelta] = context.evalThread;

	__syncthreads();

	if(tid < WARP_SIZE) {
		// Load the index of the first row encontered in this thread.
		int offset1 = tid * context.vt;
		int row1 = context.rowIndices[context.Index(offset1)];
		if(offset1 > lastTid) row1 = lastRow;

		// Load the start thread for the first row encountered in this thread.
		int startRowTid = rowStartThread_shared[row1];

		// Load the thread ranges for the tid'th row (rowDelta, not available
		// row). Find the number of store slots required for the tid'th row.
		int rowTid1 = rowStartThread_shared[tid];
		int rowTid2 = rowEndThread_shared[tid];
		bool rowValid = -1 != rowTid1;
		int rowCount = rowTid1 ? (rowTid2 - rowtid1 + 1) : 0;

		// Scan the store slots for each row. The exclusive scan is stored in
		// shared memory. The inclusive scan is retained in the register x.
		rowStartThread_shared[tid] = 0;
		volatile int* scan = rowStartThread_shared + 16;
		scan[0] = rowSlotCount;
		int x = rowSlotCount;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			int y = scan[-offset];
			x += y;
			if(i < LOG_WARP_SIZE - 1) scan[0] = x;
			else scan[0] = x - rowSlotCount;
		}

		// Pull the scan offset of the first row encountered in this eval
		// thread. Offset by the distance between this eval thread and that
		// row's starting thread (startRowTid).
		int evalStoreOffset = rowStartThread_shared[16 + row1] + tid - 
			startRowTid;

		// Cannibalize the shared memory for 







		







		int rowStart = rowStartThread_shared[tid];
		int rowBits = __ballot(-1 != rowStart);

		// Find the number of available rows preceding the tid'th row.
		int mask = bfi(0, 0xffffffff, 0, tid);
		
		int rowCount = __popc(mask & allRows);

		// Find the number of threads that this row spans.
		int rowEnd = rowEndThread_shared[tid];

		int rowSlotCount = rowEnd - rowStart + 1;
		if(-1 == rowStart) rowSlotCount = 0;


		// store flag to [x].
		// this is in range 0 - 63. each thread reads back tid and 32 + tid.
		// run a pair of ballot scans (one 32bit, one 64bit) for distance.


		// Find the 

		int firstThreadSlot = offset2 - offset1 + rowCount




	}



	





	


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

