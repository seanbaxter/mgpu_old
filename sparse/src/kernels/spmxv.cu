// Each thread will output no more than one row to tempOutput_global. However
// each row may span multiple shared memory segments (due to spanning 
// multiple threads). Therefore we need to store in sharedmem at least two
// values per thread. These will be reduced down to one value per thread on 
// output.
extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SPMXV_NAME(const uint* rowIndices_global, const uint* colIndices_global,
	 const MemType* sparseValues_global, uint numGroups,
	 ComputeType* tempOutput_global) {
	
	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;
	uint gid = NUM_WARPS * block + warp;

	// Shared memory index. Each threads needs two slots (64 per warp). For 
	// complex precision types, four slots are needed.
#ifdef USE_COMPLEX
	uint sharedOffset = 4 * WARP_SIZE * warp;
#else
	uint sharedOffset = 2 * WARP_SIZE * warp;
#endif
	uint sharedX = sharedOffset + lane;
	uint sharedY = sharedX + WARP_SIZE;

	// Load the row indices for each warp.
	if(tid < NUM_WARPS)
		sharedArray[tid] = rowIndices_global[min(numGroups - 1, gid + tid)];
	__syncthreads();
	
	uint rowIndex = sharedArray[warp];
	__syncthreads();
	
	// Break out of the kernel if the group is out of range
	if(gid >= numGroups) return;

	// offset0 is the offset of the first value of the current thread in 
	// colIndices/sparseValues. Add WARP_SIZE for each subsequent value in the 
	// thread.
	uint offset0 = WARP_SIZE * VALUES_PER_THREAD * gid + lane;
	
	// Load the column indices and sparse matrix values from global memory.
	// These are packed into the first four column indices for each thread.	
	uint colIndices[4];
	colIndices[0] = colIndices_global[offset0];
	colIndices[1] = colIndices_global[offset0 + WARP_SIZE];
	colIndices[2] = colIndices_global[offset0 + 2 * WARP_SIZE];
	colIndices[3] = colIndices_global[offset0 + 3 * WARP_SIZE];
				
	// Extract the offsets to execute the segmented scan.		
	uint scanOffset = (colIndices[0]>> 25) + sharedOffset;
	uint deltaPairX = colIndices[1]>> 26;
	uint deltaPairY = colIndices[2]>> 25;
	uint rowSumIndex = (colIndices[3]>> 25) + sharedOffset;
	
	// Although products may be up to 20 elements, it is not treated like an
	// actual array, taking that much space. The register usage of this kernel
	// should be low because, depending on how nvcc re-orders instructions to
	// increase ILP, only the last few members of products need be accessed.
	ComputeType products[VALUES_PER_THREAD];
	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {

		// Load the column index for this thread value. The first four have
		// already been loaded.
		uint offset = offset0 + WARP_SIZE * i;
		uint colIndex;
		if(i < 4) colIndex = colIndices[i];
		else colIndex = colIndices_global[offset];
		
		// Load the matrix value and up-convert into matrixValue.
		ComputeType matrixValue = ConvertUp(sparseValues_global[offset]);
		
		// Fetch the texture and up-convert into vectorValue.
		ComputeType vectorValue = FromTexture(
			tex1Dfetch(xVec_texture, 0x003fffff & colIndex));
		
		uint startFlag = FirstThreadRow & colIndex;
		uint endFlag = LastThreadRow & colIndex;

		ComputeType prev = startFlag ? products[i - 1] : Zero;

		// if(startFlag) prev = products[i - 1];
		products[i] = Add(Mul(matrixValue, vectorValue), prev);
		
		SetShared(scanOffset, products[i]);
		scanOffset += 0 != endFlag;
	}
	
	// Perform the segmented scan. Because this scan processes 64 values and
	// our warp is only 32 threads, each thread processes two slots separated by
	// 32 slots.
	ComputeType valueX = GetShared(sharedX);
	ComputeType valueY = GetShared(sharedY);
	
	// For all offsets < WARP_SIZE, we can handle the left and right halves of
	// sharedArray simultaneously, without intervening __syncthreads().
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		
		bool predX = offset <= deltaPairX;
		bool predY = offset <= deltaPairY;

		// Avoid putting multiple statements in a branch, because nvcc will
		// generate a BRA.U instruction rather than simple predication.
		if(predX) valueX = Add(valueX, GetShared(sharedX - offset));
		SetShared(sharedX, valueX);
		
		if(predY) valueY = Add(valueY, GetShared(sharedY - offset));
		SetShared(sharedY, valueY);
	}
	
	// For offset = WARP_SIZE, only handle the right half.
	bool predY = WARP_SIZE <= deltaPairY;
	if(predY) {
		valueY = Add(valueY, GetShared(sharedY - WARP_SIZE));
		SetShared(sharedY, valueY);
	}

	// Write the final row sums to tempOutput_gloabl	
	if(SerializeFlag & colIndices[1])
		// fetch the row sum from sharedArray
		tempOutput_global[rowIndex + lane] = GetShared(rowSumIndex);
}

#undef SPMXV_NAME
#undef VALUES_PER_THREAD
