// Packs partial row counts into high bits of outputIndices_global.
// y = alpha * matrix * x + beta * y
extern "C" __global__ void __launch_bounds__(128, 8)
Finalize(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, ComputeType* yVec_global,
	ComputeType alpha, ComputeType beta, uint packedSizeShift) {
	
	// We have no reduction requirements so don't bother with warp calculations.
	uint row = 128 * blockIdx.x + threadIdx.x;
	
	if(row >= numRows) return;
		
	uint index = outputIndices_global[row];
	
	uint count = index>> packedSizeShift;
	uint offset = ((1<< packedSizeShift) - 1) & index;
	ComputeType sum = Zero;
	
	for(int i = 0; i < (int)count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		sum = Add(sum, val);
	}

	sum = Mul(alpha, sum);
	if(yVec_global) {
		ComputeType y = yVec_global[row];
		sum = Add(Mul(beta, y), sum);		
	}
	yVec_global[row] = sum;
}

// Like Finalize but does not pack temp counts.
extern "C" __global__ void __launch_bounds__(128, 8)
FinalizeNoShift(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, ComputeType* yVec_global,
	ComputeType alpha, ComputeType beta) {
	
	// We have no reduction requirements so don't bother with warp calculations.
	uint row = 128 * blockIdx.x + threadIdx.x;
	
	if(row >= numRows) return;
		
	uint offset = outputIndices_global[row];
	uint next = outputIndices_global[row + 1];
	uint count = next - offset;

	ComputeType sum = Zero;
	
	for(int i = 0; i < (int)count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		sum = Add(sum, val);
	}

	sum = Mul(alpha, sum);
	if(yVec_global) {
		ComputeType y = yVec_global[row];
		sum = Add(Mul(beta, y), sum);		
	}
	yVec_global[row] = sum;
}

