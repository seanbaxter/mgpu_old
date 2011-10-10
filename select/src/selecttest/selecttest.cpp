#include "../../../util/cucpp.h"
#include "../../../inc/mgpuselect.h"
#include <vector>
#include <random>

std::tr1::mt19937 mt19937;
std::tr1::uniform_int<uint> r(0, 63);

int main(int argc, char** argv) {
	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	selectEngine_t engine;
	selectStatus_t status = selectCreateEngine(
		"../../src/cubin/ksmallest.cubin", &engine);

	int count = 500000;
	std::vector<uint> values(count);
	for(int i(0); i < count; ++i)
		values[i] = r(mt19937);

	int hist[64] = { 0 };
	for(int i(0); i < count; ++i)
		++hist[values[i]];


	DeviceMemPtr sourceDevice;
	CUresult result = context->MemAlloc(values, &sourceDevice);

	uint value;
	status = selectValue(engine, sourceDevice->Handle(), count, count / 2,
		SELECT_TYPE_UINT, &value, 0);


	selectDestroyEngine(engine);

}