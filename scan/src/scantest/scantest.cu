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
	10<< 20,
	40<< 20
};
const int NumIterations[3] = {
	1000, 
	200,
	800
};

thrust::device_ptr<uint> ThrustPtr(CuDeviceMem* mem) {
	return thrust::device_ptr<uint>((uint*)mem->Handle());
}

double Throughput(double elapsed, int count, int numIterations) {
	return (count / elapsed) * numIterations;
}

bool TestScan(int kind, int count, int numTests, int numIterations,
	scanEngine_t engine, CUDPPHandle cudpp, CuContext* context,
	double maxThroughputs[3]) {

	std::tr1::mt19937 mt19937;
	std::tr1::uniform_int<uint> r1(1, 1);
	std::tr1::uniform_int<uint> r2(0, 499);

	std::fill(maxThroughputs, maxThroughputs + 3, 0.0);

	std::vector<uint> values(count), scanRef(count), flagsHost;
	if(kind > 1) flagsHost.resize(count);
	
	uint last = 0;
	uint prevKey = 0;
	for(int i(0); i < count; ++i) {
		uint x = r1(mt19937);
		values[i] = x;

		if(kind) {
			bool head = 0 == r2(mt19937);
			if(head) {
				last = 0;
				if(1 == kind) values[i] |= 1u<< 31;
				else if(2 == kind) flagsHost[i] = 1;
				else if(3 == kind) ++prevKey;
			}
			if(3 == kind) flagsHost[i] = prevKey;
		}

		scanRef[i] = last;
		last += x;
	}

	// Allocate the device buffers.
	DeviceMemPtr valuesDevice, scanDevice, flagsDevice;
	CUresult result = context->MemAlloc(values, &valuesDevice);
	result = context->MemAlloc<uint>(count, &scanDevice);
	if(kind > 1)
		result = context->MemAlloc(flagsHost, &flagsDevice);

	// Create the CUDPP plan.
#ifdef USE_CUDPP
	CUDPPConfiguration config;
	if(0 == kind) config.algorithm = CUDPP_SCAN;
	else if(2 == kind) config.algorithm = CUDPP_SEGMENTED_SCAN;
	else cudpp = 0;
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
			timer.Start();
			// Test MGPU Scan
			for(int i(0); i < numIterations; ++i) {
				scanStatus_t status;
				if(0 == kind)
					status = scanArray(engine, valuesDevice->Handle(), 
						scanDevice->Handle(), count, 0, false);
				else if(1 == kind)
					status = scanSegmentedPacked(engine, valuesDevice->Handle(),
						scanDevice->Handle(), count, false);
				else if(2 == kind)
					status = scanSegmentedFlags(engine, valuesDevice->Handle(),
						flagsDevice->Handle(), scanDevice->Handle(), count, 
						false);
				else if(3 == kind)
					status = scanSegmentedKeys(engine, valuesDevice->Handle(),
						flagsDevice->Handle(), scanDevice->Handle(), count, 
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

			for(int i = 0; i < count; ++i)
				if(scanHost[i] != scanRef[i]) {
					printf("Error on MGPU element %d.\n", i);
					return false;
				}

			maxThroughputs[0] = std::max(maxThroughputs[0], mgpuThroughput);

			printf("MGPU: %5.2lf B/s\t\t", mgpuThroughput / 1.0e9);
		}


		////////////////////////////////////////////////////////////////////////
		// CUDPP benchmark

		double cudppThroughput = 0;

#ifdef USE_CUDPP
		if(cudpp && (0 == kind || 2 == kind)) {

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

			cudppThroughput = Throughput(timer.Stop(), count, 
				numIterations);

			// Test correctness.
			std::vector<uint> scanHost;
			result = scanDevice->ToHost(scanHost);

			if(scanHost != scanRef) {
				printf("\n\tCUDPP SORT FAILED.\n");
				return false;
			}

		}
#endif // USE_CUDPP

		maxThroughputs[1] = std::max(maxThroughputs[1], cudppThroughput);

		printf("CUDPP: %5.2lf B/s\t\t", cudppThroughput / 1.0e9);


		////////////////////////////////////////////////////////////////////////
		// Thrust benchmark

		double thrustThroughput = 0;
		if(0 == kind || 3 == kind) {
			timer.Start();
			for(int i(0); i < numIterations; ++i) {
				if(0 == kind)
					thrust::exclusive_scan(ThrustPtr(valuesDevice),
						ThrustPtr(valuesDevice) + count, ThrustPtr(scanDevice));
				else if(3 == kind)
					thrust::exclusive_scan_by_key(ThrustPtr(flagsDevice), 
						ThrustPtr(flagsDevice) + count, ThrustPtr(valuesDevice),
						ThrustPtr(scanDevice));
			}

			thrustThroughput = Throughput(timer.Stop(), count, 
				numIterations);

			// Test correctness.
			std::vector<uint> scanHost;
			result = scanDevice->ToHost(scanHost);

			if(scanHost != scanRef) {
				printf("\n\tTHRUST SORT FAILED.\n");
				return false;
			}
		}

		maxThroughputs[2] = std::max(maxThroughputs[2], thrustThroughput);

		printf("thrust: %5.2lf B/s\t\t", thrustThroughput / 1.0e9);
		
		printf("\n");
	}

#if USE_CUDPP
	if(cudpp) cudppDestroyPlan(plan);
#endif

	return true;
}

void PrintBestTime(const char* label, int test, int kind, 
	const double throughputs[4][NumSizes][3]) {

	printf("%s:\n", label);
	for(int i(0); i < NumSizes; ++i)
		printf("%5.2lf bn/s\n", throughputs[test][i][kind] / 1.0e9);
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

	double throughputs[4][NumSizes][3];
	for(int size = 0; size < NumSizes; ++size) {

		printf("\n-------------- %d elements\n", Counts[size]);
		
		printf("Global scan:\n");
		TestScan(0, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context, throughputs[0][size]);

		printf("Segmented scan (packed):\n");
		TestScan(1, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context, throughputs[1][size]);

		printf("Segmented scan (flags):\n");
		TestScan(2, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context, throughputs[2][size]);

		printf("Segmented scan (keys):\n");
		TestScan(3, Counts[size], NumTests, NumIterations[size], engine, cudpp,
			context, throughputs[3][size]);
	}

	printf("\nBest times:\n");
	PrintBestTime("MGPU scan", 0, 0, throughputs);
	PrintBestTime("CUDPP scan", 0, 1, throughputs);
	PrintBestTime("thrust scan", 0, 2, throughputs);
	PrintBestTime("MGPU seg scan (packed)", 1, 0, throughputs);
	PrintBestTime("MGPU seg scan (flags)", 2, 0, throughputs);
	PrintBestTime("CUDPP seg scan (flags)", 2, 1, throughputs);
	PrintBestTime("MGPU seg scan (keys)", 3, 0, throughputs);
	PrintBestTime("thrust seg scan (keys)", 3, 2, throughputs);
	
}

