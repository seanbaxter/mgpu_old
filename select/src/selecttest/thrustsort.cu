#include "selecttest.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

CUresult ThrustBenchmark(int iterations, int count, int k, CuContext* context,
	CuDeviceMem* randomKeys, double* elapsed, uint* element) {

	DeviceMemPtr sortData;
	CUresult result = context->MemAlloc<uint>(count, &sortData);
	if(CUDA_SUCCESS != result) return result;

	thrust::device_ptr<uint> d_vec((uint*)sortData->Handle());

	CuEventTimer timer;
	for(int i(0); i < iterations; ++i) {
		// Copy the source data into sortData.
		timer.Stop();
		thrust::device_ptr<uint> devPtr((uint*)randomKeys->Handle());
		thrust::copy(devPtr, devPtr + count, d_vec);
		timer.Start(false);

		// Sort in place with thrust.
		thrust::sort(d_vec, d_vec + count);

		// Pull the k'th smallest element.
		sortData->ToHostByte(element, 4 * k, 4);
	}
	*elapsed = timer.Stop();

	return CUDA_SUCCESS;
}

