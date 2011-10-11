#include "common.cu"

// Use 128 threads and 6 blocks per SM for 50% occupancy.
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define BLOCKS_PER_SM 6

DEVICE2 uint ConvertToUint(uint x) { return x; }
DEVICE2 uint ConvertToUint(int x) { return (uint)x + 0x80000000; }
DEVICE2 uint ConvertToUint(float x) { 
	return FloatToUint(x);
}

#include "selectcount.cu"

#include "selecthist.cu"

#include "selectstream.cu"
