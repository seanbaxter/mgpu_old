/*
Copyright (c) 2011, Sean Baxter (lightborn@gmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of SeanSparse nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Sean Baxter BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#if defined(USE_DOUBLE)
typedef double FType;
typedef double FType2;
typedef double2 CType;
typedef double2 CType2;
#define make_pair_down make_double2
#define make_pair_up make_double2
#elif defined(USE_FLOAT)
typedef float FType;
typedef float FType2;
typedef float2 CType;
typedef float2 CType2;
#define make_pair_down make_float2
#define make_pair_up make_float2
#elif defined(USE_FLOAT_DOUBLE)
typedef float FType;
typedef double FType2;
typedef float2 CType;
typedef double2 CType2;
#define make_pair_down make_float2
#define make_pair_up make_double2
#else
#error "Must define USE_FLOAT, USE_FLOAT_DOUBLE, or USE_DOUBLE"
#endif


#ifdef USE_COMPLEX
	typedef CType MemType;
	typedef CType2 ComputeType;
	#define CONVERTDOWN(c) make_pair_down((FType)c.x, (FType)c.y)
	#define CONVERTUP(c) make_pair_up((FType2)c.x, (FType2)c.y)
	#define INC(a, b) a.x += b.x; a.y += b.y
	
	#define MUL(a, b) make_pair_up(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

	// b = alpha * matrix * a + beta * b
	#define INC2(b, a, alpha, beta)						\
		b = MUL(beta, b);								\
		b.x += alpha.x * a.x - alpha.y * a.x;			\
		b.y += alpha.x * a.y + alpha.y * a.x
	#define ZERO make_pair_up(0, 0)
#else
	typedef FType MemType;
	typedef FType2 ComputeType;
	#define CONVERTDOWN(r) (FType)(r)
	#define INC(a, b) a += b
	#define INC2(y, x, alpha, beta) y = alpha * x + beta * y
	#define ZERO 0
	#define CONVERTUP(r) (FType2)(r)
#endif

typedef unsigned int uint;

// TODO: experiment using texture for tempOutput_global

// Finalize by adding the partial rows together
extern "C" __global__ void SpMxVFinalize1(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, MemType* yVec_global,
	uint packedSizeShift) {
		
	// We have no reduction requirements so don't bother with warp calculations.		
	uint gid = blockIdx.x + gridDim.x * blockIdx.y;
	uint row = blockDim.x * gid + threadIdx.x;
	
	if(row >= numRows) return;
	
	uint index = outputIndices_global[row];
	
	uint count = index>> packedSizeShift;
	uint offset = ((1<< packedSizeShift) - 1) & index;
	ComputeType sum = ZERO;
	for(uint i = 0; i < count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		INC(sum, val);
	}
	yVec_global[row] = CONVERTDOWN(sum);
}

// When the count of row fragments can't be store in outputIndices_global,
// take the different of outputIndices_global[tid + 1] and outputIndices_global[tid]
extern "C" __global__ void SpMxVFinalize2(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, MemType* yVec_global) {
	
	// We have no reduction requirements so don't bother with warp calculations.		
	uint gid = blockIdx.x + gridDim.x * blockIdx.y;
	uint row = blockDim.x * gid + threadIdx.x;
	
	if(row >= numRows) return;
	
	uint offset = outputIndices_global[row];
	uint offset2 = outputIndices_global[row + 1];
	
	uint count = offset2 - offset;
	ComputeType sum = ZERO;
	for(uint i = 0; i < count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		INC(sum, val);
	}
	yVec_global[row] = CONVERTDOWN(sum);
}


// Like SpMxVFinalize but include math for BLAS-2 increment
// y = alpha * matrix * x + beta * y
extern "C" __global__ void SpMxVFinalize3(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, MemType* yVec_global,
	uint packedSizeShift, ComputeType alpha, ComputeType beta) {
	
	// We have no reduction requirements so don't bother with warp calculations.		
	uint gid = blockIdx.x + gridDim.x * blockIdx.y;
	uint row = blockDim.x * gid + threadIdx.x;
	
	if(row >= numRows) return;
		
	uint index = outputIndices_global[row];
	MemType yMem = yVec_global[row];
	ComputeType y = CONVERTUP(yMem);
	
	uint count = index>> packedSizeShift;
	uint offset = ((1<< packedSizeShift) - 1) & index;
	ComputeType sum = ZERO;
	for(uint i = 0; i < count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		INC(sum, val);
	}
	INC2(y, sum, alpha, beta);
	yVec_global[row] = CONVERTDOWN(y);
}

// Like SpMxVFinalize but include math for BLAS-2 increment
extern "C" __global__ void SpMxVFinalize4(const ComputeType* tempOutput_global,
	const uint* outputIndices_global, uint numRows, MemType* yVec_global, 
	ComputeType alpha, ComputeType beta) {
	
	// We have no reduction requirements so don't bother with warp calculations.		
	uint gid = blockIdx.x + gridDim.x * blockIdx.y;
	uint row = blockDim.x * gid + threadIdx.x;
	
	if(row >= numRows) return;
	
	uint offset = outputIndices_global[row];
	uint offset2 = outputIndices_global[row + 1];
	MemType yMem = yVec_global[row];
	ComputeType y = CONVERTUP(yMem);
	
	uint count = offset2 - offset;
	ComputeType sum = ZERO;
	for(uint i = 0; i < count; ++i) {
		ComputeType val = tempOutput_global[offset + i];
		INC(sum, val);
	}
	INC2(y, sum, alpha, beta);
	yVec_global[row] = CONVERTDOWN(y);
}
