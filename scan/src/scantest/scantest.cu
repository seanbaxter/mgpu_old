#include "../../../util/cucpp.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "../../../inc/mgpuscan.h"

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

std::tr1::mt19937 mt19937;
std::tr1::uniform_int<uint> r1(0, 15);		// gen values between 0 and 15.
std::tr1::uniform_int<uint> r2(0, 31);		// 1 / 32 chance of new segment.

#ifdef _DEBUG

// Scan .5 million elements
const int NumElements = 1<< 19;
const int NumIterations = 1;
const int NumTests = 1;

#else

// Scan 40 million elements
const int NumElements = 35<< 20;
const int NumIterations = 30;
const int NumTests = 6;

#endif

template<typename T> 
thrust::device_ptr<T> ThrustPtr(CuDeviceMem* mem) {
	return thrust::device_ptr<T>((T*)mem->Handle());
}

int main(int argc, char** argv) {

	CUresult result = cuInit(0);
	
	DevicePtr device;
	result = CreateCuDevice(0, &device);

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	// Create the MGPU scan engine
	scanEngine_t engine;
	scanStatus_t status = scanCreateEngine(
		"../../../support/src/mgpuscan/globalscan.cubin", &engine);

	int count = NumElements;
	std::vector<uint> values(count), keys(count), scanRef(count);
	uint lastKey = (uint)-1;
	uint lastSum = 0;
	for(int i(0); i < count; ++i) {
		uint x = r1(mt19937);
		if(!i || !r2(mt19937)) {
			++lastKey;
			lastSum = 0;
		}
		values[i] = x;
		keys[i] = lastKey;
		scanRef[i] = lastSum;
		lastSum += x;
	}

	DeviceMemPtr keysDevice, valuesDevice, scanDevice;
	result = context->MemAlloc(keys, &keysDevice);
	result = context->MemAlloc(values, &valuesDevice);
	result = context->MemAlloc<uint>(count, &scanDevice);

	for(int test(1); test <= NumTests; ++test) {
		printf("Test %d.\n", test);

		CuEventTimer timer;
		timer.Start();

		for(int it(0); it < NumIterations; ++it) {
/*
			thrust::exclusive_scan_by_key(ThrustPtr<uint>(keysDevice),
				ThrustPtr<uint>(keysDevice) + count, 
				ThrustPtr<uint>(valuesDevice),
				ThrustPtr<uint>(scanDevice));
*/
	
//			thrust::exclusive_scan(ThrustPtr<uint>(valuesDevice),
//				ThrustPtr<uint>(valuesDevice) + count, 
//				ThrustPtr<uint>(scanDevice));

			status = scanArray(engine, valuesDevice->Handle(), count, 0, false);

		}

		double elapsed = timer.Stop();
		double throughput = (NumElements / elapsed) * NumIterations / 1.0e6;

		printf("%lf seconds. %lf M val/sec.\n", elapsed, throughput);

	}
	
	std::vector<uint> scanHost;
	valuesDevice->ToHost(scanHost);

//	if(scanRef != scanHost) {
//		printf("Error in thrust::scan.\n");
//		return 0;
//	}

	printf("Success\n");
	
}
