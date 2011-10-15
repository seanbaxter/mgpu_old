#include "benchmark.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// Only support 32-bit keys on key-only sorts.
void ThrustBenchmark(bool reset, int iterations, int count, CuContext* context,
	CuDeviceMem* randomKeys, CuDeviceMem* sortedKeys, double* elapsed) {

	thrust::device_vector<uint> d_vec(count);
	
	CuEventTimer timer; 
	for(int i(0); i < iterations; ++i) {
		if(!i || reset) {
			timer.Stop();

			// Copy from randomKeys to d_vec.
			thrust::device_ptr<uint> devPtr((uint*)randomKeys->Handle());
			thrust::copy(devPtr, devPtr + count, d_vec.begin());

		}
		timer.Start(false);

		// Sort in place with thrust.
		thrust::sort(d_vec.begin(), d_vec.end());
	}

	*elapsed = timer.Stop();

	// Copy from d_vec to sortedKeys.
	thrust::device_ptr<uint> devPtr((uint*)sortedKeys->Handle());
	thrust::copy(d_vec.begin(), d_vec.end(), devPtr);
}

