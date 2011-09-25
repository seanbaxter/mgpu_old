#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

// Use a 33-slot stride for shared mem transpose.
#define WARP_STRIDE (WARP_SIZE + 1)

typedef unsigned int uint;

#define DEVICE extern "C" __forceinline__ __device__ 
#define DEVICE2 __forceinline__ __device__

#define ROUND_UP(x, y) (~(y - 1) & (x + y - 1))

#include <device_functions.h>
#include <vector_functions.h>


DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


#include "globalscan.cu"

#include "segscancommon.cu"
