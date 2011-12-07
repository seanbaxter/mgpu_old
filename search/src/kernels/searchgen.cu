#define DEVICE extern "C" __device__ __forceinline__
#define DEVICE2 __device__ __forceinline__

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

// Size of a memory segment
#define SEG_SIZE 128

typedef unsigned int uint;
typedef __int64 int64;
typedef unsigned __int64 uint64;

#include <device_functions.h>
#include <vector_functions.h>

// insert the first numBits of y into x starting at bit
DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}

#include "buildtree.cu"

#include "searchtree.cu"
