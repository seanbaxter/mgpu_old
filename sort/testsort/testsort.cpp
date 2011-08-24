#include "benchmark.h"
#include <random>

std::tr1::mt19937 mt19937;

bool TestSorted(CuDeviceMem** a, CuDeviceMem** b, int numElements,
	int valueCount) {
	std::vector<uint> hostA(numElements), hostB(numElements);

	for(int i(0); i < valueCount; ++i) {
		a[i]->ToHost(&hostA[0], sizeof(uint) * numElements);
		b[i]->ToHost(&hostB[0], sizeof(uint) * numElements);

		if(a != b) return false;
	}
	return true;
}


struct Throughput {
	double elementsPerSec;
	double bytesPerSec;
	double normElementsPerSec;
	double normBytesPerSec;

	void Max(const Throughput& rhs) {
		elementsPerSec = std::max(elementsPerSec, rhs.elementsPerSec);
		bytesPerSec = std::max(bytesPerSec, rhs.bytesPerSec);
		normElementsPerSec = std::max(normElementsPerSec, 
			rhs.normElementsPerSec);
		normBytesPerSec = std::max(normBytesPerSec, rhs.normBytesPerSec);
	}
};


Throughput CalcThroughput(int numBits, int numElements, int valueCount, 
	int iterations, double elapsed) {

	Throughput throughput;
	throughput.elementsPerSec = numElements * iterations / elapsed;
	throughput.bytesPerSec = sizeof(uint) * (1 + abs(valueCount)) * 
		throughput.elementsPerSec;
	throughput.normElementsPerSec = 
		(numBits / 32.0) * throughput.elementsPerSec;
	throughput.normBytesPerSec = (numBits / 32.0) * throughput.bytesPerSec;

	return throughput;
}

// Need to sort fewer elements as the number of values increases, to fit
// everything in video memory.
#ifdef _DEBUG

const int ElementCounts[7] = {
	1234567,
//	1<< 19,
	1<< 19,
	1<< 19,
	1<< 19,
	1<< 19,
	1<< 19,
	1<< 19
};
const int NumIterations = 1;
const int NumTests = 1;

#else
/*
const int ElementCounts[7] = {
	40000000,
	27000000,
	16000000,
	12000000,
	10000000,
	80000000,
	70000000
};
const int NumIterations = 15;
const int NumTests = 5;
*/

const int ElementCounts[7] = {
	500000,
	500000,
	500000,
	500000,
	500000,
	500000,
	500000
};
const int NumIterations = 100;
const int NumTests = 5;

#endif

Throughput Thrust(CuContext* context) {
	int count = ElementCounts[0];
	Throughput thrustThroughput = { 0 };

	std::vector<uint> keysHost(count);
	std::tr1::uniform_int<uint> r(0, 0xffffffff);
	for(int i(0); i < count; ++i)
		keysHost[i] = r(mt19937);

	DeviceMemPtr keysDevice, sortedKeysDevice;
	context->MemAlloc(keysHost, &keysDevice);
	context->MemAlloc<uint>(count, &sortedKeysDevice);

	for(int test(0); test < NumTests; ++test) {
		double elapsed;
		ThrustBenchmark(true, NumIterations, count, context, keysDevice, 
			sortedKeysDevice, &elapsed);
		Throughput throughput = CalcThroughput(32, count, 0, NumIterations,
			elapsed);
		thrustThroughput.Max(throughput);
	}
	return thrustThroughput;
}

struct BenchmarkTerms {
	CuContext* context;
	sortEngine_t engine;
	int count;
	int numBits;
	int bitPass;
	int numThreads;
	int valueCount;
	int numIterations;
	int numTests;
};

bool Benchmark(BenchmarkTerms& terms, Throughput& mgpu, Throughput& b40c) {

	// Run both MGPU and B40C sorts, and compare against each other.
	int capacity = RoundUp(terms.count, 2048);

	std::vector<uint> keysHost(terms.count), valuesHost[6];
	DeviceMemPtr keysDevice, valuesDevice[6], 
		sortedKeysDevice, sortedValuesDevice[6];

	MgpuTerms mgpuTerms = { 0 };
	mgpuTerms.numBits = terms.numBits;
	mgpuTerms.iterations = terms.numIterations;
	mgpuTerms.reset = true;
	mgpuTerms.valueCount = terms.valueCount;
	mgpuTerms.count = terms.count;
	mgpuTerms.numThreads = terms.numThreads;
	mgpuTerms.bitPass = terms.bitPass;

	B40cTerms b40cTerms = { 0 };
	b40cTerms.numBits = terms.numBits;
	b40cTerms.iterations = (0 == terms.valueCount || 1 == terms.valueCount) ? 
		terms.numIterations : 1;
	b40cTerms.reset = true;
	b40cTerms.count = terms.count;
	b40cTerms.context = terms.context;

	// Generate random numbers for the keys and assign to the terms structs.
	std::tr1::uniform_int<uint> r(0, 0xffffffff>> (32 - terms.numBits));
	for(int i(0); i < terms.count; ++i)
		keysHost[i] = r(mt19937);

	// Allocate space for the random keys and sorted keys.
	terms.context->MemAlloc<uint>(capacity, &keysDevice);
	keysDevice->FromHost(keysHost);
	terms.context->MemAlloc<uint>(capacity, &sortedKeysDevice);

	mgpuTerms.randomKeys = keysDevice.get();
	mgpuTerms.sortedKeys = sortedKeysDevice.get();
	b40cTerms.randomKeys = keysDevice.get();
	b40cTerms.sortedKeys = sortedKeysDevice.get();

	if(terms.valueCount) {
		valuesHost[0].resize(terms.count);
		for(int i(0); i < terms.count; ++i)
			valuesHost[0][i] = i;

		for(int i(0); i < abs(terms.valueCount); ++i) {
			// Allocate space for the random values and sorted values.
			terms.context->MemAlloc<uint>(capacity, &valuesDevice[i]);
			terms.context->MemAlloc<uint>(capacity, &sortedValuesDevice[i]);
			valuesDevice[i]->FromHost(valuesHost[0]);

			mgpuTerms.randomVals[i] = valuesDevice[i].get();
			mgpuTerms.sortedVals[i] = sortedValuesDevice[i].get();
		}

		b40cTerms.randomVals = valuesDevice[0].get();
		b40cTerms.sortedVals = sortedValuesDevice[0].get();
	}

	double elapsed;
	Throughput throughput;
	for(int test(0); test < NumTests; ++test) {
		if(0 == terms.valueCount || 1 == terms.valueCount) {
			cudaError_t error = B40cBenchmark(b40cTerms, &elapsed);
			if(cudaSuccess != error) {
				printf("Error in B40C sort on numBits = %d.\n", terms.numBits);
				return false;
			}
			throughput = CalcThroughput(terms.numBits, terms.count, 
				terms.valueCount, terms.numIterations, elapsed);
			b40c.Max(throughput);
		}
		
		sortStatus_t status = MgpuBenchmark(mgpuTerms, terms.engine, &elapsed);
		if(SORT_STATUS_SUCCESS != status) {
			printf("Error in MGPU sort on numBits = %d: %s\n", terms.numBits,
				sortStatusString(status));
			return false;	
		}

		throughput = CalcThroughput(terms.numBits, terms.count, 
			terms.valueCount, terms.numIterations, elapsed);
		mgpu.Max(throughput);
	}

	// Read the MGPU results into host memory.
	mgpuTerms.sortedKeys->ToHost(&keysHost[0], terms.count);
	for(int i(0); i < abs(terms.valueCount); ++i) {
		valuesHost[i].resize(terms.count);
		mgpuTerms.sortedVals[i]->ToHost(&valuesHost[i][0], terms.count);
	}

	// Run the sort once on b40c to verify the results.
	b40cTerms.iterations = 1;
	cudaError_t error = B40cBenchmark(b40cTerms, &elapsed);
	if(cudaSuccess != error) {
		printf("Error in b40c sort on numBits = %d\n", terms.numBits);
		return false;
	}

	std::vector<uint> keysHost2(terms.count);
	b40cTerms.sortedKeys->ToHost(&keysHost2[0], terms.count);

	if(keysHost2 != keysHost) {
		printf("Error in sort keys on numBits = %d\n", terms.numBits);
		return false;
	}
	
	if(terms.valueCount) {
		std::vector<uint> valuesHost2(terms.count);
		b40cTerms.sortedVals->ToHost(&valuesHost2[0], terms.count);

		for(int i(0); i < abs(terms.valueCount); ++i)
			if(valuesHost[i] != valuesHost2) {
				printf("Error in sort values[%d] on numBits = %d\n", i,
					terms.numBits);
				return false;
			}
	}	

	return true;
}

bool BenchmarkBitPass(CuContext* context, sortEngine_t engine) {
	printf("Normalized throughputs for all MGPU simple output kernels.\n");
	for(int valueCount(0); valueCount <= 1; ++valueCount) {
		printf("Num values = %d\n", valueCount);
		for(int numThreads(128); numThreads <= 256; numThreads *= 2) {
			printf("Num threads = %d\n", numThreads);
			for(int bitPass(1); bitPass <= 6; ++bitPass) {
				BenchmarkTerms terms;
				terms.context = context;
				terms.engine = engine;
				terms.count = ElementCounts[abs(valueCount)];
				terms.numBits = (32 % bitPass) ? (32 - (32 % bitPass)) : 32;
				terms.bitPass = bitPass;
				terms.numThreads = numThreads;
				terms.valueCount = valueCount;
				terms.numIterations = NumIterations;
				terms.numTests = NumTests;

				Throughput mgpu = { 0 }, b40c = { 0 };
				bool success = Benchmark(terms, mgpu, b40c);
				if(!success) return false;

				printf("Bits = %d\t\t%8.3lf M/s\t%6.3lf GB/s\n", bitPass, 
					mgpu.normElementsPerSec / 1.0e6,
					mgpu.normBytesPerSec / (1<< 30));
			}
		}
		printf("\n");
	}
	return true;
}



int main(int argc, char** argv) {

	cuInit(0);
	
	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	sortEngine_t engine;
	sortStatus_t status = sortCreateEngine("..\\cubin\\", &engine);
	if(SORT_STATUS_SUCCESS != status) {
		printf("Error creating MGPU sort engine: %s\n",
			sortStatusString(status));
		return 0;
	}

	// Benchmark thrust::sort
//	Throughput thrustThroughput = Thrust(context);
//	printf("32 bit key sort with thrust::sort: %4.7lf M/s\n\n", 
//		thrustThroughput.elementsPerSec / 1.0e6);

	// Sort the keys and up to 1 value array with b40c. Sort the keys and all
	// value arrays with MGPU, and compare the results.

	// Test for all value counts.
	BenchmarkBitPass(context, engine);
/*
	for(int valueCount(0); valueCount <= 1; ++valueCount) {
		int numElements = ElementCounts[abs(valueCount)];
		switch(valueCount) {
			case -1: printf("Sorting keys/indices\n"); break;
			case 0: printf("Sorting keys\n"); break;
			case 1: printf("Sorting keys/single value\n"); break;
			default: printf("Sorting keys/%d values\n", valueCount); break;
		}
		printf("%d elements / %d iterations / %d tests.\n", numElements,
			NumIterations, NumTests);

		// Test for all bit sizes.
		for(int numBits(1); numBits <= 32; ++numBits) {

			printf("%2d bits  ", numBits);

			Throughput mgpu = { 0 }, b40c = { 0 };
			BenchmarkTerms terms = { 0 };
			terms.context = context;
			terms.engine = engine;
			terms.numBits = numBits;
			terms.count = numElements;
			terms.valueCount = valueCount;
			terms.numIterations = NumIterations;
			terms.numTests = NumTests;
			Benchmark(terms, mgpu, b40c);			

			printf("MGPU:(%8.2lf, %7.2lf) M/s",
				mgpu.elementsPerSec / 1.0e6,
				mgpu.normElementsPerSec / 1.0e6);
			if(b40c.elementsPerSec) {
				printf("  B40C:(%8.2lf, %7.2lf) M/s", 
					b40c.elementsPerSec / 1.0e6, 
					b40c.normElementsPerSec / 1.0e6);
				printf("  (%1.3lfx)", mgpu.elementsPerSec / 
					b40c.elementsPerSec);
			}
			printf("\n");
		}
		printf("\n");
	}
*/
	sortReleaseEngine(engine);
}
