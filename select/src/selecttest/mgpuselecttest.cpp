#include "selecttest.h"

selectStatus_t SelectBenchmark(int iterations, int count, int k, 
	CuContext* context, selectEngine_t engine, CuDeviceMem* randomKeys, 
	selectType_t type, double* elapsed, uint* element) {

	CuEventTimer timer;
	timer.Start();

	for(int i(0); i < iterations; ++i) {
		selectData_t data;
		data.keys = randomKeys->Handle();
		data.values = 0;
		data.count = count;
		data.bit = 0;
		data.numBits = 32;
		data.type = type;
		data.content = SELECT_CONTENT_KEYS;
		selectStatus_t status = selectItem(engine, data, k, element, 0);

		if(SELECT_STATUS_SUCCESS != status) return status;
	}

	*elapsed = timer.Stop();

	return SELECT_STATUS_SUCCESS;
}

