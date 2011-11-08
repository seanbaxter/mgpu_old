#pragma once

#include <sm_20_intrinsics.h>



#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define DEVICE extern "C" __device__ __forceinline__
#define DEVICE2 __device__ __forceinline__

typedef unsigned int uint;


// retrieve numBits bits from x starting at bit
DEVICE uint bfe(uint x, uint bit, uint numBits) {
	uint ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;
}


// insert the first numBits of y into x starting at bit
DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}
