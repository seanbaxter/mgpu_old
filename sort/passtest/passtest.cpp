

// BenchmarkBitPass benchmarks the individual bit pass speeds. The results are
// returned in a simple format that can be parsed by tablegen to create optimal
// multi-pass algorithms for sorting keys of any size.
bool BenchmarkBitPass(CuContext* context, sortEngine_t engine, 
	const char* tableSuffix) {

	printf("Normalized throughputs for all MGPU simple output kernels.\n");
	for(int valueCount(0); valueCount <= 1; ++valueCount) {
		printf("Num values = %d\n", valueCount);
		for(int numThreads(128); numThreads <= 256; numThreads *= 2) {
			printf("Num threads = %d\n", numThreads);

			printf("sort_%d_%d_%s_%s_%s", numThreads, 8, 


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

