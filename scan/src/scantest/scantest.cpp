#include "../../../util/cucpp.h"
#include "../../../inc/mgpuscan.h"

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

const int NumElements = 50<< 20;
const int NumIterations = 100;
const int NumTests = 5;

const bool TestSegmented = false;



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
		uint x = r1(mt19937);
		values[i] = x;

		if(TestSegmented) {
			bool head = 0 == r2(mt19937);
			if(head) {
				values[i] |= 1<< 31;
				last = 0;
			}
		}
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
			if(TestSegmented)
				status = scanSegmentedFlag(engine, valuesDevice->Handle(),
					scanDevice->Handle(), count, false);
			else
				status = scanArray(engine, valuesDevice->Handle(), 
					scanDevice->Handle(), count, 0, false);
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
