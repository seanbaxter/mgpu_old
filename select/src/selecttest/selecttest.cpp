#include "../../../util/cucpp.h"
#include "../../../inc/mgpuselect.h"
#include <vector>
#include <algorithm>
#include <random>


std::tr1::mt19937 mt19937;
std::tr1::uniform_int<uint> r(0, 0xffffffff);

int main(int argc, char** argv) {
	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	selectEngine_t engine;
	selectStatus_t status = selectCreateEngine(
		"../../src/cubin/select.cubin", &engine);

	int count = 40000000;
	std::vector<uint> values(count);
	for(int i(0); i < count; ++i)
		values[i] = r(mt19937);

	DeviceMemPtr sourceDevice;
	CUresult result = context->MemAlloc(values, &sourceDevice);

	CuEventTimer timer;
	timer.Start();

	int numIterations = 1500;

	// Select the item 2/3 into the sorted array.
	int k = 2 * count / 3;
	uint key;
	for(int i(0); i < numIterations; ++i) {

		selectData_t data;
		data.keys = sourceDevice->Handle();
		data.count = count;
		data.bit = 0;
		data.numBits = 32;
		data.type = SELECT_TYPE_UINT;
		data.content = SELECT_CONTENT_KEYS;

		status = selectItem(engine, data, k, &key, 0);

	//	std::sort(values.begin(), values.end());
	//	uint key2 = values[k];

	//	double split = timer.Split();
	//	printf("%d: %lf\n", i, split);
	}

	double elapsed = timer.Stop();
	double throughput = (numIterations / elapsed) * count;
	printf("GPU Throughput: 0x%08x %6.3lf M/s\n", key, throughput / 1.0e6);


	timer.Start();
	std::nth_element(values.begin(), values.begin() + k, values.end());

	elapsed = timer.Stop();
	throughput = count / elapsed;
	printf("CPU throughput: 0x%08x %6.3lf M/s\n", values[k], throughput / 1.06e6);

//	printf("MGPU Select: 0x%08x     std::sort: 0x%08x\n", key, key2);

	selectDestroyEngine(engine);

}