#include "selecttest.h"

sortStatus_t MgpuSortBenchmark(int iterations, int count, int k, 
	CuContext* context, sortEngine_t engine, CuDeviceMem* randomKeys,
	double* elapsed, uint* element) {

	MgpuSortData data;

	// Let MGPU Sort allocate all the arrays.
	sortStatus_t status = data.Alloc(engine, count, 0);
	if(SORT_STATUS_SUCCESS != status) return status;

	data.endBit = 32;

	CuEventTimer timer;
	for(int i(0); i < iterations; ++i) {
		// Copy from the source buffer to the destination.
		timer.Stop();
		randomKeys->ToDevice(data.keys[0]);
		timer.Start(false);

		// Sort and pull out the k'th elmeent.
		status = sortArray(engine, &data);
		if(SORT_STATUS_SUCCESS != status) return status;

		cuMemcpyDtoH(element, data.keys[0] + 4 * k, 4);
	}

	*elapsed = timer.Stop();

	return SORT_STATUS_SUCCESS;
}
