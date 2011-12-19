#include <device_functions.h>

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

typedef unsigned int uint;
typedef unsigned __int64 uint64;

#define DEVICE extern "C" __device__ __forceinline__
#define DEVICE2 __device__ __forceinline__

#define NUM_THREADS 512

// Same syntax as __byte_perm, but without nvcc's __byte_perm bug that masks all
// non-immediate index arguments by 0x7777.
DEVICE uint prmt(uint a, uint b, uint index) {
	uint ret;
	asm("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
	return ret;
}


////////////////////////////////////////////////////////////////////////////////
// GatherBWTKeys emits the most significant bytes for each shift of the input
// string. We read in uint32s because they are most efficient, although each 
// thread deals with values at byte-offsets. This is achieved with use of the
// prmt Fermi intrinsic.

DEVICE2 uint StoreKeys(const uint* shared, uint mask, uint& low, uint i) {

	uint high = shared[i + 1];
	uint packed = prmt(low, high, mask);
	low = high;
	return packed;
}
	
template<int NumStreams>
DEVICE2 void GatherBWTKeys(const uint* string_global, uint count,
	uint* keys1_global, uint* keys2_global, uint* keys3_global,
	uint* keys4_global, uint* keys5_global, uint* keys6_global,
	uint* indices_global) {

	const int NumLoadThreads = NUM_THREADS / 4 + WARP_SIZE;

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	
	uint gid = NUM_THREADS * block + tid;


	////////////////////////////////////////////////////////////////////////////
	// Start by loading uint32-typed values from string_global.

	__shared__ uint shared[NumLoadThreads];

	if(tid < NumLoadThreads)
		shared[tid] = string_global[NUM_THREADS * block / 4 + tid];
	__syncthreads();


	////////////////////////////////////////////////////////////////////////////
	// Use prmt to assemble 4 bytes of the desired key at a time.


	// Understand that the byte array is in an endian-neutral format. But the
	// GPU is little endian. Consider a sorted array of letters:
	// a b c d e f g h
	// We load these into low and high like (d, c, b, a), (h, g, f, e), where
	// the most-significant bytes are on the left. This is due to the way the
	// GPU loads 32-bit typed data from memory.

	// We can number these 8 characters by their prmt codes:
	// (d, c, b, a), (h, g, f, e)
	// (3, 2, 1, 0), (7, 6, 5, 4)

	// tid's that are 0 mod 4 want the tuple (a, b, c, d), with the first
	// character being most-significant. This is a big-endian reading of the
	// underlying string. 

	// We use prmt to both pack the desired bytes from low and high parts and
	// to change byte order.

	// 0 == (3 & tid): mask = 0x0123: this gathers (a, b, c, d).
	// 1 == (3 & tid): mask = 0x1234: this gathers (b, c, d, e).
	// 2 == (3 & tid): mask = 0x2345: this gathers (c, d, e, f).
	// 3 == (3 & tid): mask = 0x3456: this gathers (d, e, f, g).

	// Note these masks are gotten with the simple formula:
	// mask = 0x0123 + 0x1111 * (3 & tid).

	if(gid < count) {

		uint offset = 3 & tid;
		uint mask = 0x0123 + 0x1111 * offset;
		uint* s = shared + tid / 4;

		uint low = s[0];

		// NVCC bug:
		// Manually unroll the calls here because "dynamic" (by that I mean
		// compile-time loop-unrolled parameters) into constant buffers causes
		// uniform addressing to kick in, screwing performance. 
		if(NumStreams >= 1) keys1_global[gid] = StoreKeys(s, mask, low, 0);
		if(NumStreams >= 2) keys2_global[gid] = StoreKeys(s, mask, low, 1);
		if(NumStreams >= 3) keys3_global[gid] = StoreKeys(s, mask, low, 2);
		if(NumStreams >= 4) keys4_global[gid] = StoreKeys(s, mask, low, 3);
		if(NumStreams >= 5) keys5_global[gid] = StoreKeys(s, mask, low, 4);
		if(NumStreams >= 6) keys6_global[gid] = StoreKeys(s, mask, low, 5);

		// Initialize the index array.
		indices_global[gid] = gid;
	}
}


#define GEN_GATHER_BWT_KEYS(NumStreams)										\
	extern "C" __global__ void GatherBWTKeys_##NumStreams(					\
		const uint* string_global, uint count, uint* keys1_global,			\
		uint* keys2_global, uint* keys3_global, uint* keys4_global,			\
		uint* keys5_global, uint* keys6_global, uint* indices_global) {		\
																			\
		GatherBWTKeys<NumStreams>(string_global, count, keys1_global,		\
			keys2_global, keys3_global, keys4_global, keys5_global,			\
			keys6_global, indices_global);									\
	}

GEN_GATHER_BWT_KEYS(1)
GEN_GATHER_BWT_KEYS(2)
GEN_GATHER_BWT_KEYS(3)
GEN_GATHER_BWT_KEYS(4)
GEN_GATHER_BWT_KEYS(5)
GEN_GATHER_BWT_KEYS(6)


////////////////////////////////////////////////////////////////////////////////
// CompareBWTKeys takes the partially-sorted indices and compares the sorted
// symbols of adjacent elements to establish segments. If there are multiple
// elements in a segment, that segment requires additional sorting. This is done
// on the CPU-side using a quicksort. The quicksort uses a truncated string
// compare, which starts at the gpuKeySize symbol.

texture<uint, 1, cudaReadModeElementType> string_texture;

// nvcc seems to have problem when trying to store to an array, so we just 
// wrap it in a type and return it.
template<int NumStreams>
struct Streams {
	uint data[NumStreams];
};

template<int NumStreams>
DEVICE2 Streams<NumStreams> LoadStreams(uint index) {
	uint offset = 3 & index;
	uint mask = 0x0123 + 0x1111 * offset;

	// 4 chars per integer.
	index /= 4;

	Streams<NumStreams> streams;
	uint low = tex1Dfetch(string_texture, index);

	#pragma unroll
	for(int i = 0; i < NumStreams; ++i) {
		uint high = tex1Dfetch(string_texture, index + i + 1);
		uint packed = prmt(low, high, mask);
		streams.data[i] = packed;
		low = high;
	}
	return streams;
}

template<int NumStreams>
DEVICE2 void CompareBWTKeys(const uint* indices_global, uint count,
	uint lastMask, uint* headFlags_global) {

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint gid = NUM_THREADS * block + tid;

	const int Width = NUM_THREADS + 1;
	uint index = indices_global[gid];

	// Load in this thread's most-sig set of digits. Store in streams_shared
	// strided by NumStreams.
	__shared__ uint streams_shared[NumStreams * Width];

	if(gid < count) {
		Streams<NumStreams> streams = LoadStreams<NumStreams>(index);

		#pragma unroll
		for(int i = 0; i < NumStreams; ++i)
			streams_shared[Width * i + tid + 1] = streams.data[i];

		// On the first thread, load in the digits for the preceding key.
		if(!tid && block) {
			uint preceding = indices_global[gid - 1]; 
			Streams<NumStreams> streams2 = LoadStreams<NumStreams>(preceding);
			
			#pragma unroll
			for(int i = 0; i < NumStreams; ++i)
				streams_shared[Width * i] = streams2.data[i];
		}
	}
	__syncthreads();

	// Compare this thread's set of keys with the immediately preceding set.
	int headFlag = 0;
	#pragma unroll
	for(int i = 0; i < NumStreams; ++i) {
		uint preceding = streams_shared[Width * i + tid];
		
		// Mask out the unsorted byte for the least-significant (in big endian
		// order) register.
		if(i == NumStreams - 1) {
			preceding &= lastMask;
			streams.data[i] &= lastMask;
		}
		headFlag |= preceding != streams.data[i];
	}

	__shared__ char headFlags_shared[NUM_THREADS];
	uint* headFlagsUint = (uint*)headFlags_shared;
	headFlags_shared[tid] = (char)headFlag;
	__syncthreads();

	// Read the flags back as uints and store to global memory.
	if(tid < NUM_THREADS / 4)
		headFlags_global[(NUM_THREADS / 4) * block + tid] = headFlagsUint[tid];
}


#define GEN_COMPARE_BWT_KEYS(NumStreams)									\
	extern "C" __global__ void CompareBWTKeys_##NumStreams(					\
		const uint* indices_global, uint count, uint lastMask,				\
		uint* headFlags_global) {											\
		CompareBWTKeys<NumStreams>(indices_global, count, lastMask,			\
			headFlags_global);												\
	}

GEN_COMPARE_BWT_KEYS(1)
GEN_COMPARE_BWT_KEYS(2)
GEN_COMPARE_BWT_KEYS(3)
GEN_COMPARE_BWT_KEYS(4)
GEN_COMPARE_BWT_KEYS(5)
GEN_COMPARE_BWT_KEYS(6)
