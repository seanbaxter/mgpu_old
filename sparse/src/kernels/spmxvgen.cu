#define DEVICE extern "C" __device__ __forceinline__

// #define VALUES_PER_THREAD
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)


typedef unsigned int uint;

#ifdef USE_COMPLEX
	// Complex
	#ifdef USE_FLOAT
		#define BLOCKS_PER_SM 7
		typedef float2 MemType;
		typedef float2 ComputeType;
		typedef float2 TexType;
		typedef float SharedType;
		#define MAKE_PAIR make_float2

		DEVICE ComputeType ConvertUp(MemType m) { return m; }
		DEVICE ComputeType FromTexture(TexType t) { return t; }

	#elif defined(USE_FLOAT_DOUBLE)
		#define BLOCKS_PER_SM 6
		typedef float2 MemType;
		typedef double2 ComputeType;
		typedef uint4 TexType;
		typedef double SharedType;
		#define MAKE_PAIR make_double2

		DEVICE ComputeType ConvertUp(MemType m) { 
			return make_double2((double)m.x, (double)m.y);
		}
		DEVICE ComputeType FromTexture(TexType t) {
			return make_double2(__hiloint2double(t.y, t.x),
				__hiloint2double(t.w, t.z));
		}

	#elif defined(USE_DOUBLE)
		#define BLOCKS_PER_SM 4
		typedef double2 MemType;
		typedef double2 ComputeType;
		typedef uint4 TexType;
		typedef double SharedType;
		#define MAKE_PAIR make_double2

		DEVICE ComputeType ConvertUp(MemType m) { return m; }
		DEVICE ComputeType FromTexture(TexType t) {
			return make_double2(__hiloint2double(t.y, t.x),
				__hiloint2double(t.w, t.z));
		}
	#else
		#error "Must define USE_FLOAT, USE_FLOAT_DOUBLE, or USE_DOUBLE"
	#endif

	// Complex numbers require twice the sharedArray space. To reduce bank 
	// conflicts, put the real part at i and the complex part at 
	// 2 * WARP_SIZE + i.
	__shared__ volatile SharedType sharedArray[8 * NUM_THREADS];

	DEVICE ComputeType GetShared(int i) {
		return MAKE_PAIR(sharedArray[i], sharedArray[2 * WARP_SIZE + i]);
	}
	DEVICE void SetShared(int i, ComputeType c) {
		sharedArray[i] = c.x;
		sharedArray[2 * WARP_SIZE + i] = c.y;
	}
	DEVICE ComputeType Add(ComputeType a, ComputeType b) {
		return MAKE_PAIR(a.x + b.x, a.y + b.y);
	}
	DEVICE ComputeType Mul(ComputeType a, ComputeType b) {
		return MAKE_PAIR(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
	}

	#define Zero MAKE_PAIR(0, 0)
#else
	// Real
	#ifdef USE_FLOAT
		#define BLOCKS_PER_SM 8
		typedef float MemType;
		typedef float ComputeType;
		typedef float TexType;
		typedef float SharedType;
		
		DEVICE ComputeType ConvertUp(MemType m) { return m; }
		DEVICE ComputeType FromTexture(TexType t) { return t; }

	#elif defined(USE_FLOAT_DOUBLE)
		#define BLOCKS_PER_SM 7
		typedef float MemType;
		typedef double ComputeType;
		typedef uint2 TexType;
		typedef double SharedType;

		DEVICE ComputeType ConvertUp(MemType m) { return (double)m; }
		DEVICE ComputeType FromTexture(TexType t) {
			return __hiloint2double(t.y, t.x);
		}

	#elif defined(USE_DOUBLE)
		#define BLOCKS_PER_SM 6
		typedef double MemType;
		typedef double ComputeType;
		typedef uint2 TexType;
		typedef double SharedType;

		DEVICE ComputeType ConvertUp(MemType m) { return m; }
		DEVICE ComputeType FromTexture(TexType t) {
			return __hiloint2double(t.y, t.x);
		}

	#else
		#error "Must define USE_FLOAT, USE_FLOAT_DOUBLE, or USE_DOUBLE"
	#endif
		
			
	__shared__ volatile SharedType sharedArray[4 * NUM_THREADS];

	DEVICE ComputeType GetShared(int i) { return sharedArray[i]; }
	DEVICE void SetShared(int i, ComputeType x) { sharedArray[i] = x; }
	DEVICE ComputeType Add(ComputeType a, ComputeType b) {
		return a + b;
	}
	DEVICE ComputeType Mul(ComputeType a, ComputeType b) {
		return a * b; 
	}
	#define Zero 0
#endif



texture<TexType, cudaTextureType1D, cudaReadModeElementType> xVec_texture;

// The client must interleave data in so that coalesced reads retrieve
// data in thread order (i.e. all of tid 0's dat, then tid 1's data, etc.)

// colIndices_global flags:
// bit 23: FirstThreadRow
//		set if this is the first occurence of this row within the thread
//		If this is set, don't accumulate the preceding value in the sequential
//		scan
// bit 24: LastThreadRow
//		set if this is the last occurence of this row within the thread
//		If this is set, write this value and row index to shared memory.
//		This requires summing the number of row occurences in each thread
//		for exclusive scan allocation.
// bits 25-31: context index
//		val 0: the partial sum exclusive scan offset for this thread
//		val 1: deltaPair.x for segmented scan
//		val 2: deltaPair.y for segmented scan
//		val 3: row sum index for segmented scan


static const uint FirstThreadRow = 1<< 23;
static const uint LastThreadRow = 1<< 24;

// flag only found on the deltaX column index
static const uint SerializeFlag = 1<< 25;


#define SPMXV_NAME SpMxV_4
#define VALUES_PER_THREAD 4
#include "spmxv.cu"

#define SPMXV_NAME SpMxV_8
#define VALUES_PER_THREAD 8
#include "spmxv.cu"

#define SPMXV_NAME SpMxV_12
#define VALUES_PER_THREAD 12
#include "spmxv.cu"

#define SPMXV_NAME SpMxV_16
#define VALUES_PER_THREAD 16
#include "spmxv.cu"

#define SPMXV_NAME SpMxV_20
#define VALUES_PER_THREAD 20
#include "spmxv.cu"

#include "finalize.cu"