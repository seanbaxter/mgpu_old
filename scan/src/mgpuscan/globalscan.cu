#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_WARPS 3

#define BLOCKS_PER_SM 4

#define VALUES_PER_THREAD 8
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)

#define DEVICE extern "C" __forceinline__ __device__ 


////////////////////////////////////////////////////////////////////////////////
// Upsweep pass

DEVICE 

extern "C" __global__ __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM)
void SegScanUpsweepFlag3(const uint* valuesIn_global, uint* valuesOut_global,
	const int2* rangePairs_global) {


}