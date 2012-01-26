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

		// Mul by 4 to convert from int to byte
		uint low = bfi(0, scatter, 2, 14);
		uint high = scatter>> 14;

		StoreShifted(scattergather_shared, low, fusedKeys[2 * v]);
		StoreShifted(scattergather_shared, high, fusedKeys[2 * v + 1]);
	}
	__syncthreads();
}
	
