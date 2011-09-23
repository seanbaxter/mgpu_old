#include "../../../util/cucpp.h"
#include "../../../inc/mgpuscan.h"

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

const int NumElements = 40<< 20;//6 * 60 * 256 * 8;
const int NumIterations = 200;
const int NumTests = 4;




int main(int argc, char** argv) {


	CUresult result = cuInit(0);
	
	DevicePtr device;
	result = CreateCuDevice(0, &device);

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	// Create the MGPU scan engine
	scanEngine_t engine;
	scanStatus_t status = scanCreateEngine(
		"../../../scan/src/mgpuscan/mgpuscan.cubin", &engine);

	std::tr1::mt19937 mt19937;
	
	int count = NumElements;

	std::tr1::uniform_int<uint> r1(1, 1);
	std::tr1::uniform_int<uint> r2(0, 499);

	std::vector<uint> values(count), scanRef(count);
	uint last = 0;
	for(int i(0); i < count; ++i) {
	//	if(0 == (i % 2048)) {
	//		printf("%2d: %d\n", i / 2048, last);
	//	}
		uint x = r1(mt19937);
		values[i] = x;
		bool head = 0 == r2(mt19937);
		if(head) values[i] |= 1<< 31;
		
		if(head) last = 0;
		scanRef[i] = last;
		last += x;
	}	

	DeviceMemPtr valuesDevice, scanDevice;
	result = context->MemAlloc(values, &valuesDevice);
	result = context->MemAlloc<uint>(count, &scanDevice);

	
	//		status = scanSegmentedFlag(engine, valuesDevice->Handle(),
	//			scanDevice->Handle(), count, false);
	
	for(int test(0); test < NumTests; ++test) {

		CuEventTimer timer;
		timer.Start();

		for(int i(0); i < NumIterations; ++i) {
			status = scanSegmentedFlag(engine, valuesDevice->Handle(),
				scanDevice->Handle(), count, false);
		}

		double elapsed = timer.Stop();

		double throughput = (NumElements / elapsed) * NumIterations;
		printf("Elapsed: %2.8lf billion/s.\n", throughput / 1.0e9);
	}

	std::vector<uint> scanHost;
	result = scanDevice->ToHost(scanHost);

	for(int i(0); i < count; ++i)
		if(scanHost[i] != scanRef[i]) {
			printf("Error on item %d.\n", i);
			return 0;
		}
	printf("Success.\n");


//	scanSegmentedFlag(engine, valuesDevice-
//scanStatus_t SCANAPI scanSegmentedFlag(scanEngine_t engine, CUdeviceptr data,
//	unsigned int init, bool inclusive);


}
