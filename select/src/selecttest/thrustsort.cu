#include "selecttest.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

CUresult ThrustBenchmark(int iterations, int count, int k, CuContext* context,
	CuDeviceMem* randomKeys, selectType_t type, double* elapsed, 
	uint* element) {

	DeviceMemPtr sortData;
	CUresult result = context->MemAlloc<uint>(count, &sortData);
	if(CUDA_SUCCESS != result) return result;


	CuEventTimer timer;
	for(int i(0); i < iterations; ++i) {
		// Copy the source data into sortData.
		timer.Stop();
		randomKeys->ToDevice(sortData);
		timer.Start(false);

		// Sort in place with thrust.
		if(SELECT_TYPE_UINT == type) {
			thrust::device_ptr<uint> d_vec((uint*)sortData->Handle());
			thrust::sort(d_vec, d_vec + count);
		} else if(SELECT_TYPE_INT == type) {
			thrust::device_ptr<int> d_vec((int*)sortData->Handle());
			thrust::sort(d_vec, d_vec + count);
		} else if(SELECT_TYPE_FLOAT == type) {
			thrust::device_ptr<float> d_vec((float*)sortData->Handle());
			thrust::sort(d_vec, d_vec + count);
		}

		// Pull the k'th smallest element.
		sortData->ToHostByte(element, 4 * k, 4);
	}
	*elapsed = timer.Stop();

	return CUDA_SUCCESS;
}

