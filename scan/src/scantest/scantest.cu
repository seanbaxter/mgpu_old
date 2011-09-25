#include "../../../util/cucpp.h"
#include "../../../inc/mgpuscan.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif


// CUDPP benchmarking is optional. CUDPP requires more extensive configuration
// than thrust.
#ifdef USE_CUDPP
#include <cudpp.h>
#else
typedef void* CUDPPHandle;
#endif


const int NumSizes = 3;
const int NumTests = 5;
const int Counts[3] = {
	2<< 20,
	20<< 20,
	40<< 20
};
const int NumIterations[3] = {
	1000, 
	300,
	100
};

thrust::device_ptr<uint> ThrustPtr(CuDeviceMem* mem) {
	return thrust::device_ptr<uint>((uint*)mem->Handle());
}

double Throughput(double elapsed, int count, int numIterations) {
	return (count / elapsed) * numIterations;
}

bool TestScan(int kind, int count, int numTests, int numIterations,
	scanEngine_t engine, CUDPPHandle cudpp, CuContext* context) {

	std::tr1::mt19937 mt19937;
	std::tr1::uniform_int<uint> r1(1, 1);
	std::tr1::uniform_int<uint> r2(0, 499);

	std::vector<uint> values(count), scanRef(count), keysHost, flagsHost;
	if(2 == kind) {
		keysHost.resize(count);
		flagsHost.resize(count);
	}
	
	uint last = 0;
	uint prevKey = 0;
	for(int i(0); i < count; ++i) {
	//	if(i && (0 == (i % 4096))) {
	//		printf("%d: %d\n", i / 4096 - 1, last);
	//	}
		uint x = r1(mt19937);
		values[i] = x;

		if(1 == kind) {
			bool head = 0 == r2(mt19937);
			if(head) {
				values[i] |= 1u<< 31;
				last = 0;
			}
		} else if(2 == kind) {
			bool head = 0 == r2(mt19937);
			if(head) {
				++prevKey;
				last = 0;
			}
			keysHost[i] = prevKey;
			flagsHost[i] = (uint)head;
		}

		scanRef[i] = last;
		last += x;
	}

	// Allocate the device buffers.
	DeviceMemPtr valuesDevice, scanDevice;
	CUresult result = context->MemAlloc(values, &valuesDevice);
	result = context->MemAlloc<uint>(count, &scanDevice);

	// Create the CUDPP plan.
#ifdef USE_CUDPP
	CUDPPConfiguration config;
	if(0 == kind) config.algorithm = CUDPP_SCAN;
	else if(1 == kind) cudpp = 0;
	else if(2 == kind) config.algorithm = CUDPP_SEGMENTED_SCAN;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	CUDPPHandle plan;
	if(cudpp) {
		CUDPPResult result = cudppPlan(cudpp, &plan, config, count, 0, 0);
		if(CUDPP_SUCCESS != result) {
			printf("Could not configure CUDPP scan.\n");
			return false;
		}
	}
#endif

	for(int test(0); test < numTests; ++test) {

		CuEventTimer timer;

		////////////////////////////////////////////////////////////////////////
		// MGPU benchmark

		{
			DeviceMemPtr keysDevice;
			if(2 == kind)
				result = context->MemAlloc(keysHost, &keysDevice);

			timer.Start();
			// Test MGPU Scan
			for(int i(0); i < numIterations; ++i) {
				scanStatus_t status;
				if(0 == kind)
					status = scanArray(engine, valuesDevice->Handle(), 
						scanDevice->Handle(), count, 0, false);
				else if(1 == kind)
					status = scanSegmentedFlag(engine, valuesDevice->Handle(),
						scanDevice->Handle(), count, false);
				else if(2 == kind)
					status = scanSegmentedKeys(engine, valuesDevice->Handle(),
						keysDevice->Handle(), scanDevice->Handle(), count, 
						false);
				if(SCAN_STATUS_SUCCESS != status) {
					printf("\n\t\tMGPU error: %s.\n", scanStatusString(status));
					return false;
				}
			}
			double mgpuThroughput = Throughput(timer.Stop(), count,
				numIterations);

			// Test correctness.
			std::vector<uint> scanHost;
			result = scanDevice->ToHost(scanHost);

			if(scanHost != scanRef) {
				printf("\n\tMGPU SORT FAILED.\n");
				return false;
			}

			printf("MGPU: %2.6lf B/s\t\t", mgpuThroughput / 1.0e9);
		}


		////////////////////////////////////////////////////////////////////////
		// CUDPP benchmark

#ifdef USE_CUDPP
		if(cudpp) {
			DeviceMemPtr flagsDevice;
			if(2 == kind)
				result = context->MemAlloc(flagsHost, &flagsDevice);

			timer.Start();

			// Test MGPU Scan
			for(int i(0); i < numIterations; ++i) {
				CUDPPResult result;
				if(0 == kind)
					result = cudppScan(plan, (void*)scanDevice->Handle(),
						(const void*)valuesDevice->Handle(), count);
				else if(2 == kind)
					result = cudppSegmentedScan(plan, 
						(void*)scanDevice->Handle(), 
						(const void*)valuesDevice->Handle(),
						(const uint*)flagsDevice->Handle(), count);

				if(CUDPP_SUCCESS != result) {
					printf("\n\tCUDPP SORT FAILED.\n");
					return false;
				}
			}

			double cudppThroughput = Throughput(timer.Stop(), count, 
				numIterations);

			// Test correctness.
			std::vector<uint> scanHost;
			result = scanDevice->ToHost(scanHost);

			if(scanHost != scanRef) {
				printf("\n\tCUDPP SORT FAILED.\n");
				return false;
			}

			printf("CUDPP: %2.6lf B/s\t\t", cudppThroughput / 1.0e9);
		}
#endif // USE_CUDPP


		////////////////////////////////////////////////////////////////////////
		// Thrust benchmark

		if(1 != kind) {
			DeviceMemPtr keysDevice;
			if(2 == kind)
				result = context->MemAlloc(keysHost, &keysDevice);

			timer.Start();
			for(int i(0); i < numIterations; ++i) {
				if(0 == kind)
					thrust::exclusive_scan(ThrustPtr(valuesDevice),
						ThrustPtr(valuesDevice) + count, ThrustPtr(scanDevice));
				else if(2 == kind)
					thrust::exclusive_scan_by_key(ThrustPtr(keysDevice), 
						ThrustPtr(keysDevice) + count, ThrustPtr(valuesDevice),
						ThrustPtr(scanDevice));
			}

			double thrustThroughput = Throughput(timer.Stop(), count, 
				numIterations);

			// Test correctness.
			std::vector<uint> scanHost;
			result = scanDevice->ToHost(scanHost);

			if(scanHost != scanRef) {
				printf("\n\tTHRUST SORT FAILED.\n");
				return false;
			}

			printf("thrust: %2.6lf B/s\t\t", thrustThroughput / 1.0e9);
		}

		printf("\n");
	}

#if USE_CUDPP
	if(cudpp) cudppDestroyPlan(plan);
#endif

	return true;
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
		"../../../scan/src/mgpuscan/mgpuscan.cubin", &engine);

	// Create the CUDPP handle.
	CUDPPHandle cudpp = 0;
#ifdef USE_CUDPP
	CUDPPResult cudppResult = cudppCreate(&cudpp);
#endif

	for(int size = 2; size < NumSizes; ++size) {
		/*
		printf("Global scan -- %d elements:\n", Counts[size]);
		TestScan(0, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context);

		printf("Segmented scan (flags) -- %d elements:\n", Counts[size]);
		TestScan(1, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context);*/

		printf("Segmented scan (keys) -- %d elements:\n", Counts[size]);
		TestScan(2, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context);
	}
}

