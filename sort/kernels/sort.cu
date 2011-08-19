#define NUM_BUCKETS (1<< NUM_BITS)

#if defined(SCATTER_TRANSACTION_LIST)

// One scatter value per bucket, one gather value per bucket, and one
// transaction list per thread in the warp.
#define SCATTER_STRUCT_SIZE ROUND_UP(2 * NUM_BUCKETS + WARP_SIZE, WARP_SIZE)

// Each node in the transaction list requires 2 bytes. These are strided by the
// max number of transactions encountered. Because earlier transaction lists
// are longer than later ones by no more than 1 node, we can use dense packing
// like this without risk of error. Additionally, include at the end an offset
// and coun for each warp 
#define UNCOMPRESSED_SCATTER_SIZE (2 * MAX_TRANS(NUM_VALUES, NUM_BUCKETS))

#define SCATTER_LIST_SIZE (SCATTER_STRUCT_SIZE + \
	UNCOMPRESSED_SCATTER_SIZE + NUM_WARPS)

#define scatterList_shared SCATTER_LIST_NAME
__shared__ volatile uint scatterList_shared[SCATTER_LIST_SIZE];

#define compressedList_shared scatterList_shared
#define uncompressedList_shared (scatterList_shared + SCATTER_STRUCT_SIZE)

#elif defined(SCATTER_SIMPLE)

#define SCATTER_STRUCT_SIZE NUM_BUCKETS

#define scatterList_shared SCATTER_LIST_NAME
__shared__ volatile uint scatterList_shared[SCATTER_STRUCT_SIZE];

#define compressedList_shared scatterList_shared

#elif defined(SCATTER_INPLACE)
#endif

// Segregate the fused keys bit-by-bit to sort. When this process is completed,
// read the fused keys back into registers. Bits 6-18 of each fused key is 
// the gather index for retrieving keys. It is also used for fetching values
// from global memory.

// There are three parallel scan types:
// 1 bit scan - count the number of keys with the sort bit set. this gives us 
//   the population of bucket 0. the population of bucket 1 is found by 
//   inference.
// 2 bit scan - count the populations of four buckets. Pack into 4uint8 DWORD.
//   Perform intra-warp parallel scan on these DWORDs. Unpack into 4uint16 
//   (DWORD pair) for inter-warp multiscan phase
// 3 bit scan - count the populations of eight buckets. Pack into 8uint8 DWORD
//   pair. Perform intra-warp parallel scan on these DWORD pairs. Unpack into
//   8uint16 (DWORD quad) for inter-warp multiscan phase.


extern "C" __global__ __launch_bounds__(NUM_THREADS, NUM_BLOCKS)

#ifdef VALUE_TYPE_NONE
void SORT_FUNC(const uint* keys_global_in, const uint* bucketCodes_global,
	uint bit, uint* keys_global_out) {

#elif defined(VALUE_TYPE_INDEX)
void SORT_FUNC(const uint* keys_global_in, const uint* bucketCodes_global,
	uint bit, uint* keys_global_out, uint* index_global_out) {

#elif defined(VALUE_TYPE_SINGLE)
void SORT_FUNC(const uint* keys_global_in, const uint* bucketCodes_global,
	uint bit, uint* keys_global_out, 
	const uint* value1_global_in, uint* value1_global_out) {

#elif defined(VALUE_TYPE_MULTI)
void SORT_FUNC(const uint* keys_global_in, const uint* bucketCodes_global,
	uint bit, uint* keys_global_out, uint numValueStreams,
//	const uint* values_global_in[6], uint* values_global_out[6]
	const uint* values1_global_in, const uint* values2_global_in,
	const uint* values3_global_in, const uint* values4_global_in,
	const uint* values5_global_in, const uint* values6_global_in,
	uint* values1_global_out, uint* values2_global_out,
	uint* values3_global_out, uint* values4_global_out,
	uint* values5_global_out, uint* values6_global_out) {

#endif

	////////////////////////////////////////////////////////////////////////////
	// LOAD FUSED KEYS AND REINDEX INTO SHIFTED ORDER

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	uint* debug_global_out = keys_global_out + NUM_VALUES * block;

#if SCATTER_STRUCT_SIZE <= NUM_THREADS
	// Load the scatter (transaction) structure.
	uint globalStructOffset = SCATTER_STRUCT_SIZE * block;
	if(tid < SCATTER_STRUCT_SIZE)
		compressedList_shared[tid] = 
			bucketCodes_global[globalStructOffset + tid];
#else	
	// Load the scatter (transaction) structure.
	uint globalStructOffset = SCATTER_STRUCT_SIZE * block;
	#pragma unroll
	for(int v = 0; v < DIV_UP(SCATTER_STRUCT_SIZE, NUM_THREADS); ++v) {
		uint listOffset = NUM_THREADS * v + tid;
		if(listOffset < SCATTER_STRUCT_SIZE)
			compressedList_shared[listOffset] = 
				bucketCodes_global[globalStructOffset + listOffset];		
	}
#endif

	// Load the keys and, if sorting values, create fused keys. Store into 
	// shared mem with a WARP_SIZE + 1 stride between warp rows, so that loads
	// into thread order occur without bank conflicts.
	Values keys;
	LoadWarpValues(keys_global_in, warp, lane, block, keys);
	
#ifdef VALUE_TYPE_NONE
	// No synchronization is required because the warps are storing and loading
	// from their own intervals in shared memory.
	ScatterWarpOrder(warp, lane, true, keys);
#else
	bool fusedStridedIndex = false;
	#ifdef SCATTER_TRANSACTION_LIST
		fusedStridedIndex = true;
	#endif
	ScatterFusedWarpKeys(warp, lane, keys, bit, NUM_BITS, fusedStridedIndex);
#endif


#ifdef DETECT_SORTED
	
	// Check the early exit code. We need to sync to guarantee that all threads
	// can see the scatter flag.
	
#ifdef SCATTER_SIMPLE
	if(!tid) {
		uint scatter = compressedList_shared[0];
		*earlyExit_shared = 1 & scatter;
		compressedList_shared[0] = ~1 & scatter;		
	}
#elif defined(SCATTER_TRANSACTION_LIST)
	if(!tid) {
		uint scatter = compressedList_shared[0];
		*earlyExit_shared = 1 & scatter;
		compressedList_shared[0] = ~1 & scatter;		
	}
#endif
	__syncthreads();

#ifndef SCATTER_INPLACE
	uint isEarlyDetect = *earlyExit_shared;
#endif

	if(!isEarlyDetect) {

#endif // DETECT_SORTED

	////////////////////////////////////////////////////////////////////////////
	// SORT FUSED KEYS

	uint scanBitOffset = bit;
#ifndef IS_SORT_KEY
	scanBitOffset = 24;
#endif

#ifdef SCATTER_TRANSACTION_LIST

	bool fusedScatter = false;
	#ifndef VALUE_TYPE_NONE
		fusedScatter = true;
	#endif

#if 1 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 1, NUM_BUCKETS, compressedList_shared, 
		uncompressedList_shared, fusedScatter, debug_global_out);
#elif 2 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, NUM_BUCKETS, compressedList_shared, 
		uncompressedList_shared, fusedScatter, debug_global_out);
#elif 3 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 3, NUM_BUCKETS, compressedList_shared, 
		uncompressedList_shared, fusedScatter, debug_global_out);
#elif 4 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, NUM_BUCKETS, compressedList_shared, 
		uncompressedList_shared, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 2, 2, 0, 0, 0, fusedScatter, 
		debug_global_out);
#elif 5 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, 0, 0, 0, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 2, 3, NUM_BUCKETS, 
		compressedList_shared, uncompressedList_shared, fusedScatter,
		debug_global_out);
#elif 6 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 3, 0, 0, 0, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 3, 3, NUM_BUCKETS, 
		compressedList_shared, uncompressedList_shared, fusedScatter,
		debug_global_out);
#endif

#else

#if 1 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 1, 0, 0, 0, false, debug_global_out);
#elif 2 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, 0, 0, 0, false, debug_global_out);
#elif 3 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 3, 0, 0, 0, false, debug_global_out);
#elif 4 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, 0, 0, 0, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 2, 2, 0, 0, 0, false, debug_global_out);
#elif 5 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 2, 0, 0, 0, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 2, 3, 0, 0, 0, false, debug_global_out);
#elif 6 == NUM_BITS
	SortAndScatter(tid, scanBitOffset, 3, 0, 0, 0, false, debug_global_out);
	SortAndScatter(tid, scanBitOffset + 3, 3, 0, 0, 0, false, debug_global_out);
#endif


#endif // SCATTER_TRANSACTION_LIST

#ifdef DETECT_SORTED

	} // End of the conditional in which all the sort passes are execution.

#endif


#ifdef VALUE_TYPE_NONE
	// Pull the keys out of shared memory (they are stored with a stride of 33).
	// Examine the bits for this pass and dereferrence compressedList_shared
	// to store to global memory.

	#if defined(SCATTER_INPLACE)
		GatherBlockOrder(tid, true, keys);
		WriteKeysInplace(block, tid, keys_global_out, keys);

	#elif defined(SCATTER_SIMPLE)	
		GatherBlockOrder(tid, true, keys);
		ScatterKeysSimple(tid, keys_global_out, bit, NUM_BITS, 
			(const uint*)compressedList_shared, keys);

	#elif defined(SCATTER_TRANSACTION_LIST)

		// We'd always add lane into the scatter offset when dereferencing the
		// global out pointer. It's easiest to just add lane to it here, then 
		// save all the adds after that. Unfortunately the CUDA compiler is 
		// extremely stupid, and always wants to use ISCADD and read from the
		// global pointer from the constant buffer EVERY LOOP.
		// /*1338*/  	@P0 BRA.U 0x1358;
		// /*1340*/  	@!P0 ISCADD R6, R10, c [0x0] [0x2c], 0x2;
		// /*1348*/  	@!P0 IADD R5, R5, R6;
		// /*1350*/  	@!P0 ST [R5], R4;

		// Involving additional calculations to the pointer which involve only
		// register just make things worse. eg,
		// keys_global_out += lane + tid / 1024; (tid / 1024 is always 0):
		// /*1398*/  	@P0 BRA.U 0x13c8;
		// /*13a0*/  	@!P0 SHR.U32.W R7, R0, 0xa;
		// /*13a8*/  	@!P0 IADD R7, R10, R7;
		// /*13b0*/  	@!P0 ISCADD R7, R7, c [0x0] [0x2c], 0x2;
		// /*13b8*/  	@!P0 IADD R6, R6, R7;
		// /*13c0*/  	@!P0 ST [R6], R5;

		// Fortunately the CUDA compiler/assembler treats volatile shared memory
		// as sancrosanct. We can read a value from shared memory and shift it 
		// down to zero. Now CUDA saves the offset pointer:
		keys_global_out += lane + (uncompressedList_shared[NUM_BUCKETS]>> 31);
		// /*1280*/  	IADD R16, R2, R24;
		// /*1288*/  	@!P1 ST [R15], R6;

		// Now all operations are simply predicated, not branched, saving many 
		// cycles. Additionally, stripping the volatile qualifier from the
		// uncompressed list allows the compiler to re-order instructions to 
		// increase ILP.

		WriteCoalesced(warp, lane, NUM_BUCKETS, true, 
			(const uint*)uncompressedList_shared, keys_global_out);

	#endif

#elif defined (VALUE_TYPE_INDEX)

	#if defined(SCATTER_INPLACE)
		#error "SCATTER_INPLACE not implemented for VALUE_TYPE_SINGLE"

	#elif defined(SCATTER_SIMPLE)
	
		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		Values fusedKeys, gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);

		// Generate index values from the gather indices. As these are 
		// pre-multiplied, divide them by 4 and add the block offset.
		uint blockOffset = NUM_VALUES * block;
		Values indices;
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			indices[v] = shr_add(gather[v], 2, blockOffset);

		ScatterPairSimple(tid, keys_global_out, index_global_out, bit, NUM_BITS,
			(const uint*)compressedList_shared, keys, indices);

	#elif defined(SCATTER_TRANSACTION_LIST)
	
		// Read the fused keys from shared memory into register.
		Values scatter;
		GatherWarpOrder(warp, lane, true, scatter);
		__syncthreads();

		// Scatter the values in warp order to shared memory.
		ScatterFromIndex(scatter, true, keys);
		__syncthreads();

		// Coalesced write the keys.
		keys_global_out += lane + (compressedList_shared[0]>> 31);
		WriteCoalesced(warp, lane, NUM_BUCKETS, false, 
			(const uint*)uncompressedList_shared, keys_global_out);
		__syncthreads();

		// Generate index values from the scatter indices. As these are
		// pre-multiplied, divide them by 4 and add the block offset.
		uint laneOffset = NUM_VALUES * block + 
			VALUES_PER_THREAD * WARP_SIZE * warp + lane;
		Values indices;
		#pragma unroll
		for(int v = 0; v < VALUES_PER_THREAD; ++v)
			indices[v] = laneOffset + v * WARP_SIZE;

		ScatterFromIndex(scatter, true, indices);
		__syncthreads();

		// Coalesced write the values.
		index_global_out += lane + (compressedList_shared[0]>> 31);
		WriteCoalesced(warp, lane, NUM_BUCKETS, false, 
			(const uint*)uncompressedList_shared, index_global_out);

	#endif

#elif defined(VALUE_TYPE_SINGLE)

	#if defined(SCATTER_INPLACE)
		#error "SCATTER_INPLACE not implemented for VALUE_TYPE_SINGLE"

	#elif defined(SCATTER_SIMPLE)

		Values values;
		LoadBlockValues(value1_global_in, tid, block, values);
	
		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		Values fusedKeys, gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);
		__syncthreads();

		// Store the values to shared memory.
		ScatterBlockOrder(tid, false, values);
		// ScatterWarpOrder(warp, lane, false, values);
		__syncthreads();

		// Gather the values from shared memory.
		GatherFromIndex(gather, true, values);

		ScatterPairSimple(tid, keys_global_out, value1_global_out, bit, 
			NUM_BITS, (const uint*)compressedList_shared, keys, values);

	#elif defined(SCATTER_TRANSACTION_LIST)

		// Read the values from global memory in warp order, to match the 
		// scatter offsets collected for the keys.
		Values values;
		LoadWarpValues(value1_global_in, warp, lane, block, values);

		// Read the fused keys from shared memory into register.
		Values scatter;
		GatherWarpOrder(warp, lane, true, scatter);
		__syncthreads();

		// Scatter the values in warp order to shared memory.
		ScatterFromIndex(scatter, true, keys);
		__syncthreads();

		// Coalesced write the keys.
		keys_global_out += lane + (compressedList_shared[0]>> 31);
		WriteCoalesced(warp, lane, NUM_BUCKETS, false, 
			(const uint*)uncompressedList_shared, keys_global_out);
		__syncthreads();

		// Scater the values in warp order to shared memory.
		ScatterFromIndex(scatter, true, values);
		__syncthreads();

		// Coalesced write the values.
		value1_global_out += lane + (compressedList_shared[0]>> 31);
		WriteCoalesced(warp, lane, NUM_BUCKETS, false, 
			(const uint*)uncompressedList_shared, value1_global_out);

	#endif

#elif defined(VALUE_TYPE_MULTI)

	// We know at least two values are available with VALUE_TYPE_MULTI (or else 
	// VALUE_TYPE_SINGLE single would have been used), so we can use special
	// logic for these cases.

	#ifdef SCATTER_INPLACE
		#error "SCATTER_INPLACE not implemented for VALUE_TYPE_MULTI"

	#elif defined(SCATTER_SIMPLE)
		
		// Read the fused keys from shared memory into register and break into
		// pre-multiplied gather indices.
		Values fusedKeys, gather;
		GatherBlockOrder(tid, true, fusedKeys);
		BuildGatherFromFusedKeys(fusedKeys, gather);
		__syncthreads();

		// Store the keys to shared memory without padding.
		ScatterWarpOrder(warp, lane, false, keys);
		__syncthreads();

		// Gather the keys from shared memory.
		GatherFromIndex(gather, true, keys);
		__syncthreads();

		Values globalScatter;
		MultiScatterSimple(tid, keys_global_out, bit, NUM_BITS,
			(const uint*)compressedList_shared, keys, globalScatter);

		GlobalGatherScatter(tid, block, values1_global_in, 
			values1_global_out, gather, globalScatter);

		GlobalGatherScatter(tid, block, values2_global_in, 
			values2_global_out, gather, globalScatter);

		if(numValueStreams >= 3)
			GlobalGatherScatter(tid, block, values3_global_in, 
				values3_global_out, gather, globalScatter);

		if(numValueStreams >= 4)
			GlobalGatherScatter(tid, block, values4_global_in, 
				values4_global_out, gather, globalScatter);

		if(numValueStreams >= 5)
			GlobalGatherScatter(tid, block, values5_global_in, 
				values5_global_out, gather, globalScatter);

		if(6 == numValueStreams)
			GlobalGatherScatter(tid, block, values6_global_in, 
				values6_global_out, gather, globalScatter);

	#elif defined(SCATTER_TRANSACTION_LIST)

	#endif

#endif


}





#undef NUM_BUCKETS	
#undef SORT_FUNC
#undef NUM_BITS
#undef UNCOMPRESSED_SCATTER_SIZE
#undef SCATTER_LIST_SIZE
#undef SCATTER_LIST_NAME
#undef scatterList_shared
#undef compressedList_shared
#undef uncompressedList_shared


