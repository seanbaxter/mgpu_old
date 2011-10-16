#include "../../../util/cucpp.h"
#include "selecttest.h"
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif
std::tr1::mt19937 mt19937;

const int NumTests = 4;

const int Runs[][2] = {
	{ 125000, 1500 },
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
	selectType_t type, uint* element) {

	std::vector<uint> keys2(count);

	CuEventTimer timer;
	for(int i(0); i < iterations; ++i) {
		timer.Stop();
		std::copy(keys, keys + count, &keys2[0]);
		timer.Start(false);

		if(SELECT_TYPE_UINT == type)
			std::nth_element(&keys2[0], &keys2[0] + k, &keys2[0] + count);
		else if(SELECT_TYPE_INT == type)
			std::nth_element((int*)&keys2[0], (int*)&keys2[0] + k, 
				(int*)&keys2[0] + count);
		else if(SELECT_TYPE_FLOAT == type)
			std::nth_element((float*)&keys2[0], (float*)&keys2[0] + k, 
				(float*)&keys2[0] + count);

		*element = keys2[k];
	}
	return timer.Stop();
}


////////////////////////////////////////////////////////////////////////////////
// Time all requested algorithms for a given sequence and k.

bool BenchmarkSuite(int count, int iterations, selectType_t type, int k,
	uint* randomKeys, CuContext* context, selectEngine_t selectEngine,
	sortEngine_t sortEngine, bool doSelect, bool doMgpu, bool doThrust,
	bool doCpp) {

	// Copy the random keys to device memory.
	DeviceMemPtr keysDevice;
	CUresult result = context->ByteAlloc(sizeof(4) * count, randomKeys,
		&keysDevice);
	if(CUDA_SUCCESS != result) return false;

	uint element;

	// Benchmark select
	if(doSelect) {
		double bestSelect = DBL_MAX;
		for(int test(0); test < NumTests; ++test) {
			double elapsed;
			selectStatus_t status = SelectBenchmark(10 * iterations, count, k, 
				context, selectEngine, keysDevice, type, &elapsed, &element);
			if(SELECT_STATUS_SUCCESS != status) {
				printf("%s\n", selectStatusString(status));
				return false;
			}

			bestSelect = std::min(bestSelect, elapsed);
		}

		double selectThroughput = Throughput(bestSelect, count, 10 * iterations);
		printf("Select: %9.3lf M/s   ", selectThroughput / 1.0e6);
	}

	
	// Benchmark MGPU sort
	if(doMgpu) {
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
	}

	// Benchmark thrust sort
	if(doThrust) {
		double bestThrust = DBL_MAX;
		for(int test(0); test < NumTests; ++test) {
			double elapsed;
			result = ThrustBenchmark(iterations, count, k, context, keysDevice,
				type, &elapsed, &element);

			bestThrust = std::min(bestThrust, elapsed);
		}
		double thrustThroughput = Throughput(bestThrust, count, iterations);
		printf("thrust: %9.3lf M/s   ", thrustThroughput / 1.0e6);
	}
	
	// Bunchmark nth_element
	if(doCpp) {
		double bestCpp = DBL_MAX;
		for(int test(0); test < NumTests; ++test) {
			double elapsed = CPUBenchmark(count, DivUp(iterations, 10), k, randomKeys,
				type, &element);
			bestCpp = std::min(bestCpp, elapsed);
		}
		double cppThroughput = Throughput(bestCpp, count, 
			DivUp(iterations, 10));
		printf("cpp: %6.5lf M/s   ", cppThroughput / 1.0e6);
	}
	printf("\n");

	return true;
}


////////////////////////////////////////////////////////////////////////////////
// Benchmark uniformly random uints of different sizes.

bool BenchmarkSizes(CuContext* context, selectEngine_t selectEngine,
	sortEngine_t sortEngine) {

	std::tr1::uniform_int<uint> r(0, 0xffffffff);

	for(int i(0); i < sizeof(Runs) / 8; ++i) {
		// Generate uniformly random arrays.
		int count = Runs[i][0];
		int iterations = Runs[i][1];
		std::vector<uint> keys(count);
		for(int j(0); j < count; ++j)
			keys[j] = r(mt19937);
		
		printf("%8d: ", Runs[i][0]);
		bool success = BenchmarkSuite(count, iterations, SELECT_TYPE_UINT, 
			count / 2, &keys[0], context, selectEngine, sortEngine, true, true,
			true, true);
		if(!success) return false;
	}
	return true;
}


////////////////////////////////////////////////////////////////////////////////
// Benchmark different floating-point distributions with the same array size
// at different values of k.

std::tr1::uniform_real<float> r1(-1000, 1000);
std::tr1::uniform_real<float> r2(1, 1000000);
std::tr1::uniform_real<float> r3(-50, 50);
std::tr1::uniform_real<float> r4(-1, 1);
std::tr1::uniform_real<float> r5(-127, 127);

float Uniform1() {
	return r1(mt19937);
}
float Uniform2() {
	return r2(mt19937);
}
float Pow1() {
	float x = r4(mt19937);
	return pow(x, 7);
}
float Exp1() {
	float x = r3(mt19937);
	return powf(2.0f, x);
}
float Exp2() {
	float x = r5(mt19937);
	return powf(2.0f, x);
}
float Arctan() {
	float x = r1(mt19937);
	return atanf(x);
}
float XCos() {
	float x = r1(mt19937);
	return x * cos(x);
}


template<typename R>
bool BenchmarkDistribution(R r, const char* str, CuContext* context, 
	selectEngine_t selectEngine, int count, int iterations) {

	int NumPoints = 19;
	std::vector<float> keys(count);
	for(int i(0); i < count; ++i)
		keys[i] = //FloatToUint(r());
			r();

	printf("%20s: \n", str);

	for(int p(0); p < NumPoints; ++p) {
		int k = (int)(count * ((5 + 5 * p) / 100.0));

		printf("%5.2f%%: ", 100 * (double)k / count);
		BenchmarkSuite(count, iterations, SELECT_TYPE_FLOAT, k, (uint*)&keys[0],
			context, selectEngine, 0, true, false, false, false);
	}
	return true;
}

bool BenchmarkDistributions(CuContext* context, selectEngine_t selectEngine,
	int count, int iterations) {

	BenchmarkDistribution(Exp1, "pow(2, x) (-50 to 50)", context, selectEngine,
		count, iterations);
	BenchmarkDistribution(Exp2, "pow(2, x) (-127 to 127)", context,
		selectEngine, count, iterations);
	BenchmarkDistribution(Uniform1, "x (-1000 to 1000)", context,
		selectEngine, count, iterations);
//	BenchmarkDistribution(Uniform2, "uniform (1 to 1000000)", context,
//		selectEngine, count, iterations);
//	BenchmarkDistribution(Pow, "x^7 (-1 to 1)", context, selectEngine, count,
//		iterations);
//	BenchmarkDistribution(Arctan, "atan(x)", context, selectEngine, count,
//		iterations);
//	BenchmarkDistribution(XCos, "x * cos(x) (-1000 to 1000)", context,
//		selectEngine, count, iterations);
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

	BenchmarkSizes(context, selectEngine, sortEngine);
//	BenchmarkDistributions(context, selectEngine, 40e6, 30);


}