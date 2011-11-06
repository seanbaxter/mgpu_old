#include "common.cu"


// NOTE: complex types currently do not compile thanks to bug in float2/double2
// definitions in CUDA:
// error: no operator "=" matches these operands
//		operand types are: volatile T = T

// Must define a shared mem type copy ctor with volatile const rhs.


#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARPS_PER_BLOCK * WARP_SIZE)

#ifdef MAT_TYPE_FLOAT
	typedef float T;
	texture<float, 1, cudaReadModeElementType> xVec_texture;
	DEVICE float ReadXVec(int i) {
		return tex1Dfetch(xVec_texture, i); 
	}
	DEVICE float Add(float a, float b) { 
		return a + b;
	}
	DEVICE float MulAndAdd(float a, float b, float c) {
		return a * b + c;
	}		
	#define Zero 0.0f
	#define NUM_BLOCKS 4
#elif defined(MAT_TYPE_DOUBLE)
	typedef double T;
	texture<uint2, 1, cudaReadModeElementType> xVec_texture;
	DEVICE double ReadXVec(int i) { 
		uint2 t = tex1Dfetch(xVec_texture, i);
		return __hiloint2double(t.y, t.x);
	}
	DEVICE double Add(double a, double b) {
		return a + b;
	}
	DEVICE double MulAndAdd(double a, double b, double c) {
		return a * b + c;
	}
	#define Zero 0.0
	#define NUM_BLOCKS 3
#elif defined(MAT_TYPE_CFLOAT)
	typedef float2 T;
	texture<float2, 1, cudaReadModeElementType> xVec_texture;
	DEVICE float2 ReadXVec(int i) { 
		return tex1Dfetch(xVec_texture, i); 
	}
	DEVICE float2 Add(float2 a, float2 b) {
		return make_float2(a.x + b.x, a.y + b.y);
	}
	DEVICE float2 MulAndAdd(float2 a, float2 b, float2 c) {
		float2 product = make_float2(
			a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
		return make_float2(product.x + c.x, product.y + c.y);
	}
	#define Zero make_float2(0.0f, 0.0f)
	#define NUM_BLOCKS 3
#elif defined(MAT_TYPE_CDOUBLE)
	typedef double2 T;
	texture<uint4, 1, cudaReadModeElementType> xVec_texture;
	DEVICE double2 ReadXVec(int i) {
		uint4 t = tex1Dfetch(xVec_texture, i);
		return make_double2(
			__hiloint2double(t.y, t.x), 
			__hiloint2double(t.w, t.z));
	}
	DEVICE double2 Add(double2 a, double2 b) {
		return make_double2(a.x + b.x, a.y + b.y);
	}
	DEVICE double2 MulAndAdd(double2 a, double2 b, double2 c) {
		double2 product = make_double2(
			a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
		return make_double2(product.x + c.x, product.y + c.y);
	}
	#define Zero make_double2(0.0, 0.0)
	#define NUM_BLOCKS 3
#endif

// Flags to begin a segmented scan and to commit the scan to shared mem.
#define STORE_FLAG (1<< 25)

template<int VT>
DEVICE2 void SpMxV(const uint* rowIndices_global, const uint* colIndices_global,
	const T* sparseValues_global, T* tempOutput_global, uint numWarps) {

	__shared__ volatile T sharedSlots_shared[2 * BLOCK_SIZE];
	__shared__ volatile int outputIndices_shared[WARPS_PER_BLOCK];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;
	uint block = blockIdx.x;

	uint gid = block / WARPS_PER_BLOCK + warp;

	if(tid < WARPS_PER_BLOCK)
		outputIndices_shared[tid] = rowIndices_global[gid + tid];
	__syncthreads();

	int warpOutputIndex = outputIndices_shared[warp];

	// Break out of the kernel if the warp is out of range.
	if(gid >= numWarps) return;


	// offset0 is the offset of the first value of the current thread in 
	// colIndices/sparseValues add WARP_SIZE for each subsequent value in the
	// thread.
	uint offset0 = WARP_SIZE * VT * gid + lane;

	uint colIndices[4];
	colIndices[0] = colIndices_global[offset0 + 0 * WARP_SIZE];
	colIndices[1] = colIndices_global[offset0 + 1 * WARP_SIZE];
	colIndices[2] = colIndices_global[offset0 + 2 * WARP_SIZE];
	colIndices[3] = colIndices_global[offset0 + 3 * WARP_SIZE];

	uint sharedOffset = 2 * WARP_SIZE * warp;
	uint scanOffset = (colIndices[0]>> 26) + sharedOffset;
	uint deltaPairX = colIndices[1]>> 27;
	uint deltaPairY = colIndices[2]>> 26;
	uint rowSumIndex = (colIndices[3]>> 26) + sharedOffset;
	uint storeToGlobal = (1<< 26) & colIndices[1];


	////////////////////////////////////////////////////////////////////////////
	// Run the sequential part of the sparse matrix * vector.

	T sum = Zero;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		uint colIndex;
		if(i < 4) colIndex = colIndices[i];
		else colIndex = colIndices_global[offset0 + i * WARP_SIZE];

		T sparseValue = sparseValues_global[offset0 + i * WARP_SIZE];
		T xValue = ReadXVec(0x007fffff & colIndex);

		sum = MulAndAdd(sparseValue, xValue, sum);
		int store = (i == (VT - 1)) || (STORE_FLAG & colIndex);

		// Write the 
		if(store) sharedSlots_shared[scanOffset] = sum;
		if(store) ++scanOffset;
		if(store) sum = Zero;
	}


	////////////////////////////////////////////////////////////////////////////
	// Intra-warp parallel scan.

	volatile T* data = sharedSlots_shared + sharedOffset;

	T valueX = data[0];
	T valueY = data[WARP_SIZE];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;

		bool predX = offset <= deltaPairX;
		bool predY = offset <= deltaPairY;

		T leftX = data[-offset];
		T leftY = data[WARP_SIZE - offset];

		if(predX) valueX = Add(valueX, leftX);
		if(predY) valueY = Add(valueY, leftY);

		data[0] = valueX;
		data[WARP_SIZE] = valueY;
	}

	if(WARP_SIZE <= deltaPairY)
		valueY = Add(valueY, valueX);
	data[WARP_SIZE] = valueY;


	////////////////////////////////////////////////////////////////////////////
	// Store the temp dot products to tempOutput_global.
	
	if(storeToGlobal)
		tempOutput_global[warpOutputIndex + lane] = 
			sharedSlots_shared[sharedOffset + rowSumIndex];
}


#define GEN_SPMXV(count)													\
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, NUM_BLOCKS)				\
void SpMxV_##count(															\
	const uint* rowIndices_global, const uint* colIndices_global,			\
	const T* sparseValues_global, T* tempOutput_global, uint numWarps) {	\
																			\
	SpMxV<count>(rowIndices_global, colIndices_global, sparseValues_global,	\
		tempOutput_global, numWarps);										\
}


GEN_SPMXV(4)
GEN_SPMXV(6)
GEN_SPMXV(8)
GEN_SPMXV(10)
GEN_SPMXV(12)
GEN_SPMXV(16)
GEN_SPMXV(20)
