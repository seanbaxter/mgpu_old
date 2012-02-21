#include "benchmark.h"

sortStatus_t MgpuBenchmark(MgpuTerms& terms, sortEngine_t engine, 
	double* elapsed) {

	MgpuSortData data;

	// Put the output arrays as the input arrays into the terms.. This saves
	// some device memory.
	data.AttachKey(terms.sortedKeys->Handle(), 1);
	for(int i(0); i < abs(terms.valueCount); ++i)
		data.AttachVal(i, terms.sortedVals[i]->Handle(), 1);

	sortStatus_t status = data.Alloc(engine, terms.count, terms.valueCount);
	if(SORT_STATUS_SUCCESS != status) return status;

	data.endBit = terms.numBits;

	CuEventTimer timer;
	for(int i(0); i < terms.iterations; ++i) {
		if(!i || terms.reset) {
			timer.Stop();
			uint bytes = sizeof(uint) * terms.count;
			terms.randomKeys->ToDevice(0, data.keys[0], bytes);
			for(int i(0); i < terms.valueCount; ++i)
				terms.randomVals[i]->ToDevice(0, data.values[i][0], bytes);
		}

		timer.Start(false);

		if(!terms.bitPass)
			status = sortArray(engine, &data);
		else
			status = sortArrayEx(engine, &data, terms.numThreads,
				terms.valuesPerThread, terms.bitPass, terms.useTransList);
		if(SORT_STATUS_SUCCESS != status) return status;
	}
	*elapsed = timer.Stop();

	// Copy the results back to terms.sortedKeys and terms.sortedVals.
	if(!data.parity) {
		uint bytes = sizeof(uint) * terms.count;
		CUresult result = terms.sortedKeys->FromDevice(0, data.keys[0], bytes);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
		for(int i(0); i < abs(terms.valueCount); ++i) {
			result = terms.sortedVals[i]->FromDevice(0, data.values[i][0],
				bytes);
			if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
		}
	}

	return SORT_STATUS_SUCCESS;
}

