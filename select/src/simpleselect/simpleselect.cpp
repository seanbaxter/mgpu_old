#include "../../../inc/mgpuselect.h"
#include "../../../util/cucpp.h"

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

std::tr1::mt19937 mt19937;

int main(int argc, char** argv) {
	// Initialize CUDA driver API.
	CUresult result = cuInit(0);

	DevicePtr device;
	result = CreateCuDevice(0, &device);
	if(CUDA_SUCCESS != result) {
		printf("Could not create device.\n");
		return 0;
	}

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	// Load the MGPU Select engine from select.cubin.
	selectEngine_t engine;
	selectStatus_t status = selectCreateEngine("../../src/cubin/select.cubin", 
		&engine);
	if(SELECT_STATUS_SUCCESS != status) {
		printf("%s\n", selectStatusString(status));
		return 0;
	}

	// Generate a million uniform random distribution of 24-bit uints.
	std::tr1::uniform_int<uint> r(0, 0x00ffffff);

	int count = 1000000;
	std::vector<uint> keysHost(count);
	for(int i(0); i < count; ++i)
		keysHost[i] = r(mt19937);

	DeviceMemPtr keysDevice;
	result = context->MemAlloc(keysHost, &keysDevice);

	// Fill out selectData_t.
	selectData_t data;
	data.keys = keysDevice->Handle();
	data.values = 0;
	data.count = count;
	data.bit = 0;				// least sig bit is 0
	data.numBits = 24;			// 24 bits
	data.type = SELECT_TYPE_UINT;
	data.content = SELECT_CONTENT_KEYS;

	uint key;
	int k = count / 2;
	status = selectItem(engine, data, k, &key, 0);
	if(SELECT_STATUS_SUCCESS != status) {
		printf("%s\n", selectStatusString(status));
		return 0;
	}

	printf("Median value is %d.\n", key);
}

