#pragma once

#include "sortlocal.cu"

// Read fused keys from shared memory, scan, and scatter the fused keys into
// strided shared memory. 

DEVICE void SortAndScatter(uint tid, Values fusedKeys, uint bit, uint numBits,
	uint numThreads, bool loadKeysFromArray, bool storeStrided, 
	uint* scattergather_shared, uint* scratch_shared, uint* debug_global) {

	uint packed[4];
	Values digits;

	// Load the keys from 
	if(loadKeysFromArray) {	
		volatile uint* threadData = scattergather_shared + 
			StridedThreadOrder(tid * VALUES_PER_THREAD);
			
		#pragma unroll
		for(int v = 0; v < 8; ++v)
			fusedKeys[v] = threadData[v];
	}

	// Extract the digits.
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD; ++v)
		digits[v] = bfe(fusedKeys[v], bit, numBits);

	// Perform the scans for the actual sort.
	FindScatterIndices(tid, digits, numBits, numThreads, scratch_shared, packed,
		debug_global);

	// Unpack the scatter indices.
	#pragma unroll
	for(int v = 0; v < VALUES_PER_THREAD / 2; ++v) {
		uint scatter = packed[v];
		
		if(storeStrided)
			//	indexPacked += (0xffe0ffe0 & scatter)>> 5;
			scatter = shr_add(0xffe0ffe0 & scatter, 5, scatter);

	//	scatter<<= 2;			// mul by 4 to convert from int to byte
	//	uint low = 0x0000ffff & scatter;
	//	uint high = scatter>> 16;

		uint low = bfi(0, scatter, 2, 14);
		uint high = scatter>> 14;


	//	uint low = 0xffff & scatter;
	//	uint high = scatter>> 16;

		StoreShifted(scattergather_shared, low, fusedKeys[2 * v]);
		StoreShifted(scattergather_shared, high, fusedKeys[2 * v + 1]);

	//	debug_global[VALUES_PER_THREAD * tid + 2 * v + 0] = low;
	//	debug_global[VALUES_PER_THREAD * tid + 2 * v + 1] = high;

	//	scattergather_shared[VALUES_PER_THREAD * tid + 2 * v + 0] = low;
	//	scattergather_shared[VALUES_PER_THREAD * tid + 2 * v + 1] = high;
	}
	__syncthreads();
}
	
