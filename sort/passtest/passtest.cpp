

#include "../../inc/mgpusort.hpp"
#include "../../util/cucpp.h"
#include <vector>

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



// BenchmarkBitPass benchmarks the individual bit pass speeds. The results are
// returned in a simple format that can be parsed by tablegen to create optimal
// multi-pass algorithms for sorting keys of any size.
bool BenchmarkBitPass(CuContext* context, sortEngine_t engine, 
	const int* testSizes, int numIterations, int numTests,
	const char* tableSuffix) {

	for(int valueCount(0); valueCount <= 1; ++valueCount) {
		for(int numThreads(128); numThreads <= 256; numThreads *= 2) {
			
			// Formulate a table name like sort_128_8_key_simple_table
			printf("sort_%d_8_", numThreads, 8);
			switch(valueCount) {
				case -1: printf("index_"); break;
				case 0: printf("key_"); break;
				case 1: printf("single_"); break;
				default: printf("multi_%d_", valueCount); break;
			}
			// Only benchmark simple storage for now
			printf("simple_");

			printf("%s\n", tableSuffix);

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

				printf("%7.3lf\n", mgpu.normElementsPerSec / 1.0e6);
			}
		}
		printf("\n");
	}
	return true;
}

int main(int argc, char** argv) {

	cuInit(0);

	DevicePtr device;
	CUresult result = CreateCuDevice(0, &device);

	ContextPtr context;




}
