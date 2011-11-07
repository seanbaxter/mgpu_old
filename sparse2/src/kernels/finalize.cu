
// Like Finalize but does not pack temp counts.
extern "C" __global__ void Finalize(const T* tempOutput_global,
	const uint* rowIndices_global, uint numRows, T* yVec_global,
	T alpha, T beta, int useBeta) {

	__shared__ int volatile shared[BLOCK_SIZE + 1];
	
	// We have no reduction requirements so don't bother with warp calculations.
	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint row = BLOCK_SIZE * block + tid;

	if(row <= numRows) 
		shared[tid] = rowIndices_global[row];
	if((BLOCK_SIZE * (block + 1) <= numRows) && !tid)
		shared[BLOCK_SIZE] = rowIndices_global[row + BLOCK_SIZE];
	__syncthreads();
	
	if(row >= numRows) return;
		
	uint offset = shared[tid];
	uint next = shared[tid + 1];
	uint count = next - offset;

	T sum = Zero;
	
	for(int i = 0; i < (int)count; ++i) {
		T val = tempOutput_global[offset + i];
		sum = Add(sum, val);
	}

	sum = Mul(alpha, sum);
	if(useBeta) {
		T y = yVec_global[row];
		sum = MulAndAdd(beta, y, sum);
	}
	yVec_global[row] = sum;
}

