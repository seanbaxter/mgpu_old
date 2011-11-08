#include "common.cu"


#ifdef MAT_TYPE_FLOAT
	typedef uint T;
#elif defined(MAT_TYPE_DOUBLE)
	typedef uint2 T;
#elif defined(MAT_TYPE_CFLOAT)
	typedef uint2 T;
#elif defined(MAT_TYPE_CDOUBLE)
	typedef uint4 T;
#endif



DEVICE2 void Zero(uint& x) { x = 0; }
DEVICE2 void Zero(uint2& x) { x = make_uint2(0, 0); }
DEVICE2 void Zero(uint4& x) { x = make_uint4(0, 0, 0, 0); }


// Flags to begin a segmented scan and to commit the scan to shared mem.
#define STORE_FLAG (1<< 25)

// Each thread initializes pointers dynamically.
struct Context {
	// Shared memory arrays. rowIndices and colIndices hold at least 
	int* rowIndices;
	int* colIndices;
	T* values;
	int* transposeBuffer;

	uint numValues;
	uint vt;
	int available;

	uint sharedScatter;
	uint sharedGather;

	// tid handles the value associated with (evalThread, evalValue) in the 
	// evaluation kernel.
	uint evalThread;
	
	// Help in converting strided index to thread index. If valuesPerThread is
	// odd, we can simply dereference into shared memory without any bank 
	// conflicts. If valuesPerThread is a power of 2, we use a stride of 33 
	// (add offset / WARP_SIZE). If valuesPerThread is even but not a power of
	// two, we'll have to eat a two-way serialization penalty, but can use the
	// native stride of 32.
	bool stridedIndex;
};

DEVICE int Index(const Context& context, int offset) {
	if(context.stridedIndex) offset += offset / WARP_SIZE;
	return offset;
}


DEVICE int TransposeLocal(int x, Context& context) {
	context.transposeBuffer[context.sharedScatter] = x;
	__syncthreads();
	x = context.transposeBuffer[context.sharedGather];
	__syncthreads();
	return x;
}

// Returns the number of values consumed.
DEVICE int ProcessRowIndices(uint tid, Context& context, 
	int*& colIndices_global, T*& values_global, int*& outputIndices_global) {

	volatile int* tempSpace_shared = (volatile int*)context.transposeBuffer;
	
	volatile int* first_shared = tempSpace_shared;
	volatile int* last_shared = tempSpace_shared + 32;
	volatile int* firstValueCache_shared = tempSpace_shared + 64;
	volatile int& lastTid_shared = tempSpace_shared[96];


	////////////////////////////////////////////////////////////////////////////
	// LOAD ROW INDICES AND FIND RANGE OF ACTUAL VALUES TO CONSUME IN THIS
	// BLOCK.
	// Values from rows firstRow through firstRow + 32 (exclusive) are
	// "available" and streamed to the encoded block. 

	// Clear the rowStartThread and rowEndThread arrays. Any index other than 
	// -1 indicates that the row in question is in the block.
	if(tid < 2 * WARP_SIZE)
		first_shared[tid] = -1;
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
	int nextTest = (tid + 1 < context.available) && (nextRow < endRow);

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
	if(prevRow != threadRow) first_shared[rowDelta] = context.evalThread;
	if(threadRow != nextRow) last_shared[rowDelta] = context.evalThread;
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// CALCULATE SPECIAL OFFSETS FOR FIRST FOUR VALUES OF EACH EVAL THREAD.

	if(tid < WARP_SIZE) {
		// Load the index of the first row encontered in this thread.
		int offset1 = tid * context.vt;
		int row1 = context.rowIndices[Index(context, offset1)];
		if(offset1 > lastTid) row1 = lastRow;

		// Load the start thread for the first row encountered in this thread.
		int startRowTid = first_shared[row1];

		// Load the thread ranges for the tid'th row (rowDelta, not available
		// row). Find the number of store slots required for the tid'th row.
		int rowTid1 = first_shared[tid];
		int rowTid2 = last_shared[tid];
		bool rowValid = -1 != rowTid1;
		int rowSlotCount = rowTid1 ? (rowTid2 - rowTid1 + 1) : 0;

		// Scan the store slots for each row. The exclusive scan is stored in
		// shared memory. The inclusive scan is retained in the register x.
		tempSpace_shared[tid] = 0;
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
			(63 - tid - __clz(scanStartX));

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
		tempSpace_shared[96 + target] = lastRowSlot<< 26;


		// Store the global row indices.
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
	*colIndices_global = colIndex;
	colIndices_global += context.numValues;

	
	////////////////////////////////////////////////////////////////////////////
	// LOAD VALUES AND TRANSPOSE AND STORE.
	
	T value;
	Zero(value);
	if(tid < lastTid) value = context.values[context.sharedGather];

	*values_global = value;
	values_global += context.numValues;

	return lastTid + 1;
}


DEVICE void MoveToFront(uint tid, int consumed, Context& context) {

	int index1 = Index(context, tid);
	int index2 = Index(context, context.numValues + tid);

	if(consumed < context.available) {
		// Move all the values left by consumed slots.
		int tidRow1 = context.rowIndices[tid];
		int tidCol1 = context.colIndices[index1];
		T val1 = context.values[index1];

		int tidRow2, tidCol2;
		T val2;
		if(tid < WARP_SIZE) {
			tidRow2 = context.rowIndices[context.numValues + tid];
			tidCol2 = context.colIndices[index2];
			val2 = context.values[index2];
		}
		__syncthreads();

		if(tid < WARP_SIZE) {
			int i = context.numValues + tid - consumed;
			int index = Index(context, i);			
			context.rowIndices[i] = tidRow2;
			context.colIndices[index] = tidCol2;
			context.values[index] = val2;
		}
		if(tid >= consumed) {
			int i = tid - consumed;
			int index = Index(context, i);
			context.rowIndices[i] = tidRow1;
			context.colIndices[index] = tidCol1;
			context.values[index] = val1;
		}
	}
	context.available -= consumed;
}

DEVICE void RepopulateSharedRows(uint tid, int numValues, int& remaining,
	const int*& rowIndices_global, const int*& colIndices_global,
	const T*& values_global, Context& context) {

	// Load up to numValues from rowIndices_global. Always load a multiple of
	// 32 values for coalescing.
	int remaining2 = ~(WARP_SIZE - 1) & context.available;
	int count = min(numValues - remaining2, remaining);

	if(tid < count) {
		int i = context.available + tid;
		int index = Index(context, i);
		context.rowIndices[i] = rowIndices_global[tid];
		context.colIndices[index] = colIndices_global[tid];
		context.values[index] = values_global[tid];
	}
	__syncthreads();
	
	rowIndices_global += count;
	colIndices_global += count;
	values_global += count;
	context.available += count;
}

template<int VT>
DEVICE2 void MatrixEncode(
	const int* rowIndices_global, const int* colIndices_global,
	const T* sparseValues_global, const int2* rangePairs_global,
	const int4* groupInfo_global, int* colIndicesOut_global,
	T* sparseValuesOut_global, int* rowIndicesOut_global,
	int* outputIndicesOut_global, int height) {

	const int Count = VT * WARP_SIZE;
	const int Stride = ((Count - 1) & Count) ? 32 : 33;

	// Keep the row indices with a stride of 32 for fast random access.
	__shared__ int rowIndices_shared[Count + WARP_SIZE];

	// Keep the col indices and values with a stride (33 for pow2 VT) for fast
	// transpose from strided order to thread order.
	__shared__ int colIndices_shared[Stride * (VT + 1)];
	__shared__ T values_shared[Stride * (VT + 1)];
	__shared__ int transpose_shared[Stride * VT];
	
	uint tid = threadIdx.x;	
	uint block = blockIdx.x;
	int2 rangePair = rangePairs_global[block];
	int remaining = rangePair.y - rangePair.x;

	Context context;
	context.available = 0;
	context.rowIndices = rowIndices_shared;
	context.colIndices = colIndices_shared;
	context.values = values_shared;
	context.transposeBuffer = transpose_shared;

	context.numValues = Count;
	context.vt = VT;

	// This should be the only divide in the kernel.
	context.evalThread = tid / VT;



	while(remaining) {
		RepopulateSharedRows(tid, Count, remaining, rowIndices_global, 
			colIndices_global, sparseValues_global, context);

		int consumed = ProcessRowIndices(tid, context, colIndicesOut_global,
			sparseValuesOut_global, outputIndicesOut_global);

		MoveToFront(tid, consumed, context);
	}

	// Cap out the last index.
}


#define DEFINE_ENCODE(vt)													\
extern "C" __global__ void MatrixEncode_##count(							\
	const int* rowIndices_global, const int* colIndices_global,				\
	const T* sparseValues_global, const int2* rangePairs_global,			\
	const int4* groupInfo_global, int* colIndicesOut_global,				\
	T* sparseValuesOut_global, int* rowIndicesOut_global,					\
	int* outputIndicesOut_global, int height) {								\
																			\
	MatrixEncode<vt>(rowIndices_global, colIndices_global,					\
		sparseValues_global, rangePairs_global, groupInfo_global,			\
		colIndicesOut_global, sparseValuesOut_global, rowIndicesOut_global, \
		outputIndicesOut_global, height);									\
}

DEFINE_ENCODE(4)


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

