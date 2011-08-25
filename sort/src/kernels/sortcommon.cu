// Include just once - this code is the same no matter NUM_BUCKETS

#define MAX_BITS 6
#define MAX_BUCKETS (1<< MAX_BITS)

typedef uint Values[VALUES_PER_THREAD];

#define BIT(bit) (1<< bit)
#define MASK(high, low) (((31 == high) ? 0 : (1u<< (high + 1))) - (1<< low))
#define MASKSHIFT(high, low, x) ((MASK(high, low) & x) >> low)


// Characterize the shared memory usage of the 3-bit multi-scan (the most
// memory-intensive bit of code). All other scans fall within this envelope.

// There are three kinds of shared space here:
// 1) SCATTER-GATHER SPACE
// This is shared memory that sits at scattegather_shared. It can only be used
// when there is no contention for it from other threads expecting the scatted
// keys or values to be there. Using this destroys the scattergather_shared 
// array, but each thread should have its fused keys in register. It may only be
// used inside a __syncthreads pair (in that pair no fused keys are gathered or
// scattered) in the scan routines. Re-using this shared memory allows for 
// plenty of space for transaction list processing.

// 2) SCRATCH SPACE
// This is shared memory that comes after scattergather_shared 
// (SCATTER_GATHER_SPACE size) but before the compressend and expanded 
// scatter/transaction lists. It is used to store predIncs for each warp after
// LoadFusedKeys but before the multi-scan.

// 3) SCATTER SPACE
// This shared memory holds compressed scatter list and optionally uncompressed
// scatter list. Because the scatter space varies depending on the number of
// bits in the sort, it is defined as a separate shared memory array per sort
// kernel. This may lead to higher occupancy for smaller radix digit sorts.


// Use a stride of 33 values between rows.
#define SCATTER_GATHER_SIZE (NUM_VALUES + NUM_VALUES / WARP_SIZE)

// Reduction scratch space should be conserved to allow sufficient space for 
// transaction lists and even caching of keys for fused key sorts.
// The 3-bit multi-scan stores two values per thread. These are strided with an
// extra slot for each WARP_SIZE of elements. Additionally, 64 slots are
// required to hold the sequential scan results.
#define REDUCTION_SCRATCH_SIZE_SCAN3 \
	((2 * (NUM_THREADS + (NUM_THREADS / WARP_SIZE))) + 64)
#define REDUCTION_SCRATCH_SIZE REDUCTION_SCRATCH_SIZE_SCAN3

// In addition to REDUCTION_SCRATCH_SIZE, reserve one word for early exit code.
#define SCRATCH_SIZE (REDUCTION_SCRATCH_SIZE + 1)	

__shared__ volatile uint scattergather_shared[SCATTER_GATHER_SIZE];
__shared__ volatile uint scratch_shared[SCRATCH_SIZE];

#define reduction_shared scratch_shared
#define earlyExit_shared (scratch_shared + REDUCTION_SCRATCH_SIZE)

// Define SCATTER_SPACE_SIZE in the sort file, per kernel.

// The maximum number of optimal transactions per configuration is unknown to 
// me. Intuition tells me it's the total number of data warps 
// (NUM_VALUES / WARP_SIZE) plus the number of buckets, but this count is often
// exceeded. Running bucketmap 200,000 iterations per configuration gives these
// results:

// warps = 16  buckets = 2 target = 18    actual = 19   delta = 1
// warps = 16  buckets = 4 target = 20    actual = 23   delta = 3
// warps = 16  buckets = 8 target = 24    actual = 28   delta = 4
// warps = 16  buckets = 16 target = 32    actual = 37   delta = 5
// warps = 16  buckets = 32 target = 48    actual = 59   delta = 11
// warps = 16  buckets = 64 target = 80    actual = 94   delta = 14
// warps = 16  buckets = 128 target = 144    actual = 157   delta = 13
// warps = 32  buckets = 2 target = 34    actual = 35   delta = 1
// warps = 32  buckets = 4 target = 36    actual = 39   delta = 3
// warps = 32  buckets = 8 target = 40    actual = 45   delta = 5
// warps = 32  buckets = 16 target = 48    actual = 54   delta = 6
// warps = 32  buckets = 32 target = 64    actual = 72   delta = 8
// warps = 32  buckets = 64 target = 96    actual = 111   delta = 15
// warps = 32  buckets = 128 target = 160    actual = 179   delta = 19
// warps = 64  buckets = 2 target = 66    actual = 67   delta = 1
// warps = 64  buckets = 4 target = 68    actual = 71   delta = 3
// warps = 64  buckets = 8 target = 72    actual = 77   delta = 5
// warps = 64  buckets = 16 target = 80    actual = 86   delta = 6
// warps = 64  buckets = 32 target = 96    actual = 105   delta = 9
// warps = 64  buckets = 64 target = 128    actual = 138   delta = 10
// warps = 64  buckets = 128 target = 192    actual = 213   delta = 21

// For 16 warps of data (64 threads) with 32 buckets, we have encountered
// 59 transactions. This exceeds my target estimate by 18.6%. To be well
// on the safe side, 25% extra space is reserved. It seems astronomically
// improbable that this would not be enough space. It can however be increased
// at any time, at the risk of reduced occupancy.

#define MAX_TRANS(values, buckets) (5 * (values / WARP_SIZE + buckets) / 4)



#if defined(VALUE_TYPE_INDEX) || defined(VALUE_TYPE_SINGLE)
	#define IS_SORT_PAIR
#endif

////////////////////////////////////////////////////////////////////////////////
// Load global keys and values.

// LoadWarpValues loads so that each warp has values that are contiguous in
// global memory. Rather than having a stride of NUM_THREADS, the values are
// strided with WARP_SIZE. This is advantageous, as it lets us scatter to shared
// memory, the read into register arrays in thread-order, and process in 
// thread-order, without requiring __syncthreads, as the warps are independent.

// Read from global memory into warp order.
DEVICE void LoadWarpValues(const uint* values_global, uint warp, uint lane, 
	uint block, Values values) {

	uint threadStart = NUM_VALUES * block + 
		VALUES_PER_THREAD * WARP_SIZE * warp + lane;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = values_global[threadStart + v * WARP_SIZE];
}

// Read from global memory into block order.
DEVICE void LoadBlockValues(const uint* values_global, uint tid, uint block, 
	Values values) {
	uint threadStart = NUM_VALUES * block + tid;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = values_global[threadStart + v * NUM_THREADS];
}


////////////////////////////////////////////////////////////////////////////////
// Gather and scatter to shared memory.

// If strided is true, move keys/fused keys to shared memory with a 33 slot 
// stride between rows. This allows conflict-free gather into thread order.

DEVICE void GatherWarpOrder(uint warp, uint lane, bool strided, Values values) {
	// Each warp deposits VALUES_PER_THREAD rows to shared mem. The stride 
	// between rows is WARP_SIZE + 1. This enables shared loads with no bank
	// conflicts.
	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = scattergather_shared[sharedStart + stride * v];
}
DEVICE void ScatterWarpOrder(uint warp, uint lane, bool strided,
	const Values values) {
	// Each warp deposits VALUES_PER_THREAD rows to shared mem. The stride
	// between rows is WARP_SIZE + 1. This enables shared loads with no bank
	// conflicts.
	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;
	
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		scattergather_shared[sharedStart + stride * v] = values[v];
}


DEVICE void ScatterBlockOrder(uint tid, bool strided, const Values values) {
	volatile uint* shared = scattergather_shared + tid;
	if(strided) shared += tid / WARP_SIZE;

	uint stride = strided ? 
		(NUM_THREADS + NUM_THREADS / WARP_SIZE) : NUM_THREADS;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		shared[stride * v] = values[v];
}
DEVICE void GatherBlockOrder(uint tid, bool strided, Values values) {
	volatile uint* shared = scattergather_shared + tid;
	if(strided) shared += tid / WARP_SIZE;

	uint stride = strided ?
		(NUM_THREADS + NUM_THREADS / WARP_SIZE) : NUM_THREADS;

	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		values[v] = shared[stride * v];
}


DEVICE void GatherFromIndex(const Values gather, bool premultiplied, 
	Values data) {

	if(premultiplied) {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			data[v] = *(uint*)((char*)(scattergather_shared) + gather[v]);
	} else {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			data[v] = scattergather_shared[gather[v]];
	}
}

DEVICE void ScatterFromIndex(const Values scatter, bool premultiplied, 
	const Values data) {

	if(premultiplied) {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			*(uint*)((char*)(scattergather_shared) + scatter[v]) = data[v];
	} else {
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			scattergather_shared[scatter[v]] = data[v];
	}
}


////////////////////////////////////////////////////////////////////////////////
// Fused key support

// Build fused keys by packing the radix digits into 31:24 and the index within
// the block into 23:0.

DEVICE void ScatterFusedWarpKeys(uint warp, uint lane, const Values keys, 
	uint bitOffset, uint numBits, bool strided) {

	uint stride = strided ? (WARP_SIZE + 1) : WARP_SIZE;
	uint sharedStart = stride * VALUES_PER_THREAD * warp + lane;

	Values fusedKeys;
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint key = bfe(keys[v], bitOffset, numBits);
		uint index = 4 * (sharedStart + stride * v);
		fusedKeys[v] = index + (key<< 24);
	}
	ScatterWarpOrder(warp, lane, true, fusedKeys);
}

DEVICE void ScatterFusedBlockKeys(uint tid, const Values keys, uint bitOffset, 
	uint numBits, bool strided) {

	uint stride, sharedStart;
	if(strided) {
		stride = NUM_THREADS + (NUM_THREADS / WARP_SIZE);
		sharedStart = tid + (tid / WARP_SIZE);
	} else {
		stride = NUM_THREADS;
		sharedStart = tid;
	}

	Values fusedKeys;
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v) {
		uint key = bfe(keys[v], bitOffset, numBits);
		uint index = 4 * (sharedStart + stride * v);
		fusedKeys[v] = index + (key<< 24);
	}
	ScatterBlockOrder(tid, true, fusedKeys);
}

DEVICE void BuildGatherFromFusedKeys(const Values fusedKeys, Values scatter) {
	#pragma unroll 
	for(int v = 0; v < VALUES_PER_THREAD; ++v) 
		scatter[v] = 0x00ffffff & fusedKeys[v];
}
