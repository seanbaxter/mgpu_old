#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

typedef unsigned int uint;

#define DEVICE extern "C" __forceinline__ __device__ 
#define DEVICE2 __forceinline__ __device__

#include "globalseg.cu"