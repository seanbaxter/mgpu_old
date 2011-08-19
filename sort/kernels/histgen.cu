
// Defines NUM_COUNT_WARPS and NUM_HIST_WARPS
#include "params.cu"
#define NUM_WARPS NUM_HIST_WARPS

#include "common.cu"

#define MAX_BITS 6
#define MAX_BUCKETS (1<< MAX_BITS)

// #define INCLUDE_TRANSACTION_LIST
#define NUM_TRANSACTION_LISTS WARP_SIZE
#define LOG_NUM_TRANSACTION_LISTS LOG_WARP_SIZE

typedef unsigned int uint;
typedef unsigned short uint16;

// Histogram 1
__shared__ volatile uint hist_shared1[2 * NUM_THREADS];

// Histogram 2
__shared__ volatile uint blockScan_shared2[8 * NUM_THREADS];
__shared__ volatile uint bucketTotals_shared2[MAX_BUCKETS];

// Histogram 3
// Because we may index before the beginning of a shared memory array during
// scan, we need to be certain that we aren't actually going into negative 
// shared memory addresses. Simply including a 
//		__shared__ volatile uint prefix_shared3[WARP_SIZE];
// term at the start will not remedy this, as if the array is unreferenced, the
// compiler will drop the array and move the next array to the start. To 
// circumvent this, define the shared memory once, and use macros to establish 
// individual array names within that shared memory block.
__shared__ volatile uint hist3_shared3[10 * NUM_THREADS];
#define columnScan_shared3 (hist3_shared3 + WARP_SIZE)
#define scatterScan_shared3 (columnScan_shared3 + 2 * NUM_THREADS)
#define listScan_shared3 (scatterScan_shared3 + NUM_THREADS)
#define gatherScan_shared3 (listScan_shared3 + NUM_THREADS)
#define transOffset_shared3 (gatherScan_shared3 + NUM_THREADS)
#define bucketCodes_shared3 (transOffset_shared3 + NUM_THREADS)

#define DEVICE extern "C" __device__ __forceinline

// Return the number of transactions required to scatter keyCount words starting
// at offset scatter.
DEVICE uint NumBucketTransactions(uint scatter, uint keyCount) {
	uint start = ~(WARP_SIZE - 1) & scatter;
	uint end = ~(WARP_SIZE - 1) & (scatter + keyCount + WARP_SIZE - 1);
	uint count = (end - start) / WARP_SIZE;
	return keyCount ? count : 0;
}

// Divides the total evenly by the number of lists. When total is not a multiple
// of NUM_TRANSACTION_LISTS, lists at the beginning will have an extra 
// transaction.
DEVICE uint NumListTransactions(uint list, uint total) {
	uint count = total / NUM_TRANSACTION_LISTS;
	count += list < ((NUM_TRANSACTION_LISTS - 1) & total);
	return count;
}

// Returns the first transaction this list makes.
DEVICE uint ListTransactionOffset(uint list, uint total) {
	uint offset = list * (total / NUM_TRANSACTION_LISTS);
	offset += min(list, (NUM_TRANSACTION_LISTS - 1) & total);
	return offset;
}

// Return -1 if no warp starts in this bucket. Otherwise return the warp index.
// Fermi hardware lacks integer division. the ptxas-generated code uses a
// multi-instruction floating-point sequence that uses more registers than are
// required, and causes spill.
DEVICE uint MapBucketToList(uint transaction, uint count, uint total) {
	
	// n is minimum number of transactions per list. All lists to the right of
	// m have this number of transactions.
	uint n = total / NUM_TRANSACTION_LISTS;

	// m is number of lists with n + 1 transactions
	uint m = (NUM_TRANSACTION_LISTS - 1) & total;

	// Split is the transaction offset for list = m. Before this offset, each
	// list has n + 1 transactions. After this offset, each list has n
	// transactions.
	uint split = n * m + m;

	// We essentially evaluate list = transaction * NUM_LISTS / n, to map a
	// bucket into the first list that maps into it, but compensate for m != 0,
	// i.e. non-uniformly sized lists. If this bucket's transaction comes at or
	// after the split, then each list in that region has n transactions. 
	// Otherwise the lists have n + 1 transactions. It is imperative to round
	// up when dividing, so that a bucket refers to the first list that STARTS
	// in that bucket. Do that by adding the number of transactions per list in
	// that region minus one to the numerator.
	uint num, denom;
	if(transaction >= split) 
		num = transaction - split + n - 1, 
		denom = n;
	else 
		num = transaction + n, 
		denom = n + 1;

	uint list = num / denom;

	// Because we subtracting split out from transaction, add the number of
	// lists skipped back in.
	if(transaction >= split) list += m;
	
	// Get the starting transaction for this list. If it falls in the current
	// bucket, then we have a hit. Otherwise return 0xffffffff to prevent the 
	// caller from using this mapping.
	uint offset = list * n + min(list, m);
	return (offset < transaction + count) ? list : 0xffffffff;
}


DEVICE uint GetNextOccupiedBucket(uint bucket, uint count, int numBits,
	volatile uint* scan) {

	uint next = count ? bucket : 0xffffffff;
	scan[0] = next;
	#pragma unroll
	for(int i = 0; i < numBits; ++i) {
		uint offset = 1<< i;
		uint y = scan[offset];

		// Update if the next pointer is -1 and we aren't reading off the end.
		if(bucket < ((1<< numBits) - offset) && (0xffffffff == next))
			next = y;
		scan[0] = next;
	}
	
	// return the subsequent bucket, if we aren't the last bucket.
	if(next < (1<< numBits) - 1) next = scan[1];
	return next;
}


// Scan from right into left. This can be function is parameterized to be used
// either to compute scatters (the same bucket over multiple sort blocks) or
// gathers (different buckets within a sort block) dependending on pred and
// stride.
DEVICE uint HistParallelScan(uint pred, uint stride, uint x, int loops, 
	volatile uint* scan) {

	uint sum = x;
	scan[0] = x;
	#pragma unroll
	for(int i = 0; i < loops; ++i) {
		uint offset = stride<< i;
		uint y = scan[-offset];
		if(pred >= offset) x += y;
		scan[0] = x;
	}
	return x - sum;
}



#define NUM_BITS 1
#define HISTOGRAM_FUNC1 HistogramReduce_1_1
#define HISTOGRAM_FUNC2 HistogramReduce_2_1
#define HISTOGRAM_FUNC3 HistogramReduce_3_1
#define ComputeGatherStruct ComputeGatherStruct1
#define ComputeTransactionList ComputeTransactionList1
#include "histogram.cu"

#define NUM_BITS 2
#define HISTOGRAM_FUNC1 HistogramReduce_1_2
#define HISTOGRAM_FUNC2 HistogramReduce_2_2
#define HISTOGRAM_FUNC3 HistogramReduce_3_2
#define ComputeGatherStruct ComputeGatherStruct2
#define ComputeTransactionList ComputeTransactionList2
#include "histogram.cu"

#define NUM_BITS 3
#define HISTOGRAM_FUNC1 HistogramReduce_1_3
#define HISTOGRAM_FUNC2 HistogramReduce_2_3
#define HISTOGRAM_FUNC3 HistogramReduce_3_3
#define ComputeGatherStruct ComputeGatherStruct3
#define ComputeTransactionList ComputeTransactionList3
#include "histogram.cu"

#define NUM_BITS 4
#define HISTOGRAM_FUNC1 HistogramReduce_1_4
#define HISTOGRAM_FUNC2 HistogramReduce_2_4
#define HISTOGRAM_FUNC3 HistogramReduce_3_4
#define ComputeGatherStruct ComputeGatherStruct4
#define ComputeTransactionList ComputeTransactionList4
#include "histogram.cu"

#define NUM_BITS 5
#define HISTOGRAM_FUNC1 HistogramReduce_1_5
#define HISTOGRAM_FUNC2 HistogramReduce_2_5
#define HISTOGRAM_FUNC3 HistogramReduce_3_5
#define ComputeGatherStruct ComputeGatherStruct5
#define ComputeTransactionList ComputeTransactionList5
#include "histogram.cu"

#define NUM_BITS 6
#define HISTOGRAM_FUNC1 HistogramReduce_1_6
#define HISTOGRAM_FUNC2 HistogramReduce_2_6
#define HISTOGRAM_FUNC3 HistogramReduce_3_6
#include "histogram.cu"

