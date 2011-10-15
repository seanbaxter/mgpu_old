#include "../../../util/cucpp.h"
#include "selecttest.h"
#include <vector>
#include <algorithm>
#include <random>


std::tr1::mt19937 mt19937;

const int NumTests = 4;

const int Runs[][2] = {
	{ 250000, 1000 },
	{ 500000, 700 },
	{ 1000000, 500 },
	{ 2000000, 300 },
	{ 5000000, 200 },
	{ 10000000, 120 },
	{ 15000000, 100 },
	{ 20000000, 80 },
	{ 25000000, 65 },
	{ 30000000, 50 },
	{ 35000000, 45 },
	{ 40000000, 40 },
	{ 45000000, 30 },
	{ 50000000, 25 }
};

double Throughput(double elapsed, int count, int iterations) {
	return (count / elapsed) * iterations;
}

double CPUBenchmark(int count, int iterations, int k, const uint* keys, 
	uint* element) {

	std::vector<uint> keys2(count);

	CuEventTimer timer;
	for(int i(0); i < iterations; ++i) {
		timer.Stop();
		std::copy(keys, keys + count, &keys2[0]);
		timer.Start(false);

		std::nth_element(&keys2[0], &keys2[0] + k, &keys2[0] + count);
		*element = keys2[k];
	}
	return timer.Stop();
}

bool Benchmark(int count, int iterations, CuContext* context, 
	selectEngine_t selectEngine, sortEngine_t sortEngine) {

	std::tr1::uniform_int<uint> r(0, 0xffffffff);
	std::vector<uint> keys(count);

	for(int i(0); i < count; ++i)
		keys[i] = r(mt19937);

	DeviceMemPtr keysDevice;
	CUresult result = context->MemAlloc(keys, &keysDevice);
	if(CUDA_SUCCESS != result) return false;

	uint element;
	int k = count / 2;

	// Benchmark select
	double bestSelect = DBL_MAX;
	for(int test(0); test < NumTests; ++test) {
		double elapsed;
		selectStatus_t status = SelectBenchmark(10 * iterations, count, k, 
			context, selectEngine, keysDevice, &elapsed, &element);
		if(SELECT_STATUS_SUCCESS != status) {
			printf("%s\n", selectStatusString(status));
			return false;
		}

		bestSelect = std::min(bestSelect, elapsed);
	}

	double selectThroughput = Throughput(bestSelect, count, 10 * iterations);
	printf("Select: %9.3lf M/s   ", selectThroughput / 1.0e6);

	
	// Benchmark MGPU sort
	double bestMgpu = DBL_MAX;
	for(int test(0); test < NumTests; ++test) {
		double elapsed;
		sortStatus_t status = MgpuSortBenchmark(iterations, count, k, context,
			sortEngine, keysDevice, &elapsed, &element);
		if(SORT_STATUS_SUCCESS != status) {
			printf("%s\n", sortStatusString(status));
			return false;
		}

		bestMgpu = std::min(bestMgpu, elapsed);
	}
	double mgpuThroughput = Throughput(bestMgpu, count, iterations);
	printf("MGPU: %9.3lf M/s   ", mgpuThroughput / 1.0e6);


	// Benchmark thrust sort
	double bestThrust = DBL_MAX;
	for(int test(0); test < NumTests; ++test) {
		double elapsed;
		result = ThrustBenchmark(iterations, count, k, context, keysDevice,
			&elapsed, &element);

		bestThrust = std::min(bestThrust, elapsed);
	}
	double thrustThroughput = Throughput(bestThrust, count, iterations);
	printf("thrust: %9.3lf M/s   ", thrustThroughput / 1.0e6);

	
	// Bunchmark nth_element
	double bestCpp = DBL_MAX;
	for(int test(0); test < NumTests; ++test) {
		double elapsed = CPUBenchmark(count, iterations / 10, k, &keys[0],
			&element);
		bestCpp = std::min(bestCpp, elapsed);
	}
	double cppThroughput = Throughput(bestCpp, count, iterations / 10);
	printf("cpp: %6.5lf M/s", cppThroughput / 1.0e6);


	printf("\n");

	return true;
}






int main(int argc, char** argv) {
	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	// Load the MGPU Select engine.
	selectEngine_t selectEngine;
	selectStatus_t selectStatus = selectCreateEngine(
		"../../src/cubin/select.cubin", &selectEngine);
	if(SELECT_STATUS_SUCCESS != selectStatus) {
		printf("%s\n", selectStatusString(selectStatus));
		return 0;
	}

	// Load the MGPU Sort engine.
	sortEngine_t sortEngine;
	sortStatus_t sortStatus = sortCreateEngine(
		"../../../sort/src/cubin/", &sortEngine);
	if(SORT_STATUS_SUCCESS != sortStatus) {
		printf("%s\n", sortStatusString(sortStatus));
		return 0;
	}


	for(int i(0); i < sizeof(Runs) / 8; ++i) {
		printf("%8d: ", Runs[i][0]);
		Benchmark(Runs[i][0], Runs[i][1], context, selectEngine, sortEngine);
	}

}