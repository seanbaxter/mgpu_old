#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define DEVICE extern "C" __device__ __forceinline__ 
#define DEVICE2 __device__ __forceinline__ 

typedef unsigned int uint;

#define MAT_TYPE_FLOAT

#ifdef MAT_TYPE_FLOAT
	#define T uint
#elif defined(MAT_TYPE_DOUBLE)
	#define T uint2
#elif defined(MAT_TYPE_CFLOAT)
	#define T uint2
#elif defined(MAT_TYPE_CDOUBLE)
	#define T uint4
#endif


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

DEVICE2 void Zero(uint& x) { x = 0; }
DEVICE2 void Zero(uint2& x) { x = make_uint2(0, 0); }
DEVICE2 void Zero(uint4& x) { x = make_uint4(0, 0, 0, 0); }


// Flags to begin a segmented scan and to commit the scan to shared mem.
const uint STORE_FLAG = 1<< 25;

// Each thread initializes pointers dynamically.
struct ThreadContext {
	// Shared memory arrays.
	uint* rowIndices;
	uint* colIndices;
	T* values;
	uint* transposeBuffer;

	// Global memory arrays.
	uint* colIndices_global;
	T* values_global;


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


uint TransposeLocal(uint x, ThreadContext context) {
	context.transposeBuffer[context.sharedScatter] = x;
	__syncthreads();
	x = transposeBuffer[context.sharedGather];
	__syncthreads();
	return x;
}


DEVICE void ProcessRowIndices(uint tid, ThreadContext context) {
	volatile int* tempSpace_shared = (volatile int*)context.transposeBuffer;
	
	volatile int* first_shared = tempSpace_shared;
	volatile int* last_shared = tempSpace_shared + 32;
	volatile int* firstValueCache_shared = tempSpace_shared + 64;
	volatile int& lastTid_share = tempSpace_shared[96];


	////////////////////////////////////////////////////////////////////////////
	// LOAD ROW INDICES AND FIND RANGE OF ACTUAL VALUES TO CONSUME IN THIS
	// BLOCK.
	// Values from rows firstRow through firstRow + 32 (exclusive) are
	// "available" and streamed to the encoded block. 

	// Clear the rowStartThread and rowEndThread arrays. Any index other than 
	// -1 indicates that the row in question is in the block.
	if(tid < 2 * WARP_SIZE)
		rowStartThread_shared[tid] = -1;
	__syncthreads();

	// Get the first row, tid's row, and the row at tid + 1.
	int firstRow = context.rowIndices[0];
	int threadRow = context.rowIndices[tid];
	int nextRow = context.rowIndices[tid + 1];

	// endRow is one past the last row that can possibly be encountered in this
	// block.
	int endRow = firstRow + WARP_SIZE;

	// These values are true if the thread's value is in the block.
	int threadTest = (tid < context.available) && (threadRow < endRow);
	int nextTest = (tid + 1 < available) && (nextRow < endRow);

	// The last in-range thread writes to lastTid_shared.
	if(threadTest != nextTest) lastTid_shared = tid;
	__syncthreads();

	// Retrieve the tid of the last element in the block and its row index.
	uint lastTid = lastTid_shared;
	int lastRow = context.rowIndices[lastTid];

	// If threadRow or nextRow is out-of-bounds, set it to lastRow.
	int prevRow = context.rowIndices[tid - 1];
	prevRow = min(prevRow, lastRow);
	threadRow = min(threadRow, lastRow);
	nextRow = min(nextRow, lastRow);

	int rowDelta = threadRow - firstRow;

	// If this is the first value of this row in the block, store to
	// rowStart_shared.
	if(prev != threadRow) first_shared[rowDelta] = context.evalThread;
	if(threadRow != nextRow) last_shared[rowDelta] = context.evalThread;
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// CALCULATE SPECIAL OFFSETS FOR FIRST FOUR VALUES OF EACH EVAL THREAD.

	if(tid < WARP_SIZE) {
		// Load the index of the first row encontered in this thread.
		int offset1 = tid * context.vt;
		int row1 = context.rowIndices[context.Index(offset1)];
		if(offset1 > lastTid) row1 = lastRow;

		// Load the start thread for the first row encountered in this thread.
		int startRowTid = first_shared[row1];

		// Load the thread ranges for the tid'th row (rowDelta, not available
		// row). Find the number of store slots required for the tid'th row.
		int rowTid1 = first_shared[tid];
		int rowTid2 = last_shared[tid];
		bool rowValid = -1 != rowTid1;
		int rowCount = rowTid1 ? (rowTid2 - rowtid1 + 1) : 0;

		// Scan the store slots for each row. The exclusive scan is stored in
		// shared memory. The inclusive scan is retained in the register x.
		rowStartThread_shared[tid] = 0;
		volatile int* scan = tempSpace_shared + 16;
		scan[0] = rowSlotCount;
		int incScan = rowSlotCount;
		int excScan;
		#pragma unroll
		for(int i = 0; i < LOG_WARP_SIZE; ++i) {
			int offset = 1<< i;
			int y = scan[-offset];
			incScan += y;
			if(i < LOG_WARP_SIZE - 1) scan[0] = incScan;
			else {
				excScan = incScan - rowSlotCount;
				scan[0] = excScan;
			}
		}

		// THREAD INDEX A: thread scan offset.
		// Pull the scan offset of the first row encountered in this eval
		// thread. Offset by the distance between this eval thread and that
		// row's starting thread (startRowTid).
		int evalStoreOffset = tempSpace_shared[16 + row1] + tid - 
			startRowTid;

		// THREAD INDEX B, C: segmented scan distance.
		// Store a bit at the start of each available row in the scan array.
		// Read these back at tid and 32 + tid and use __ballot and __clz to
		// find the start of the sgement (that is, the most-significant set bit
		// at or before the current position).
		tempSpace_shared[tid] = 0;
		tempSpace_shared[32 + tid] = 0;
		if(rowValid) tempSpace_shared[excScan] = 1;

		uint scanStartX = __ballot(tempSpace_shared[tid]);
		uint scanStartY = __ballot(tempSpace_shared[WARP_SIZE + tid]);

		uint mask = 0xffffffff>> (31 - tid);
		uint distanceX = 31 - tid - __clz(mask & scanStartX);
		uint distanceY = scanStartY ? 
			(31 - tid - __clz(mask & scanStartY)) : 
			(63 - tid - __clz(scanStart));

		// Count the total number of rows in the black.
		uint rowBits = __ballot(rowValid);
		uint precedingRows = __popc(bfi(0, 0xffffffff, 0, tid) & rowBits);
		uint numRows = __popc(rowBits);

		// THREAD INDEX D: last scan slot for each available row. 
		// If this is a valid row, store the last index for it. Otherwise, store
		// zero.
		uint lastRowSlot = rowValid ? (incScan - 1) : 0;
		uint target = rowValid ?
			precedingRows :
			(tid - precedingRows + numRows);

		// Store the four special terms.
		tempSpace_shared[tid] = evalStoreOffset<< 26;
		tempSpace_shared[32 + tid] = (distanceX<< 27) | ((tid < numRows)<< 26);
		tempSpace_shared[64 + tid] = distanceY<< 26;
		if(rowValid) tempSpace_shared[96 + precedingRows] = lastRowSlot<< 26;
	}
	__syncthreads();

	int storeBit = (threadRow != nextRow) ? STORE_FLAG : 0;

	////////////////////////////////////////////////////////////////////////////
	// LOAD COL INDICES AND ATTACH SPECIAL OFFSETS AND STORE FLAGS.

	uint colIndex = 0;
	if(tid < lastTid) colIndex = context.rowIndices[tid];
	if(threadRow != nextRow) colIndex |= STORE_FLAG;

	// Transpose to put the decorated column index into strided order, suitable
	// for storing to global memory.
	colIndex = TransposeLocal(colIndex, context);

	// Apply the special indices for the first four values for each eval thread.
	if(tid < 4 * WARP_SIZE) colIndex |= tempSpace_shared[tid];

	// Store the decorated column indices to global memory.
	*context.colIndices_global = colIndex;

	
	////////////////////////////////////////////////////////////////////////////
	// LOAD VALUES AND TRANSPOSE AND STORE.
	
	T value;
	Zero(value);
	if(tid < lastTid) value = context.values[context.sharedGather];

	*context.values_global = value;


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

