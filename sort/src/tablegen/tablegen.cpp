// Tablegen takes in throughputs for all six sort passes and finds optimal
// sort combinations for keys of sizes 1 through 32 bits.

#include <cstdio>
#include <algorithm>

const int MaxBits = 7;


// holds number instances of bit passes for 1, 2, 3, 4, 5, 6, 7 NUM_BITS.
double Throughput(const double* passThroughputs, int targetBits, 
	const int* counts) {

	double inv = 0;
	for(int i(0); i < MaxBits; ++i)
		inv += (i + 1) * counts[i] / passThroughputs[i];
	return targetBits / inv;
}

// Iterate over all passes from greatest number of bits (6) to fewest number
// (1). When we've exceeded the target number of bits, break.

void FindOptimalPath(const double* passThroughputs, int* counts, int numBits,
	int targetBits, int* optCounts, double& optThroughput) {
	
	// Sum up the number of bits already taken up in the sort.
	int encountered = 0;
	for(int j(MaxBits); j > numBits; --j)
		encountered += j * counts[j - 1];
	
	// Recursively test all the combinations until we exceed the number of target bits.
	if(1 == numBits) {
		int passCount = targetBits - encountered;
		counts[numBits - 1] = passCount;
		double t = Throughput(passThroughputs, targetBits, counts);
		if(t > optThroughput) {
			std::copy(counts, counts + MaxBits, optCounts);
			optThroughput = t;
		} 
	} else {
		int passCount = 0;
		while(encountered + numBits * passCount <= targetBits) {
			counts[numBits - 1] = passCount;
			FindOptimalPath(passThroughputs, counts, numBits - 1, targetBits, 
				optCounts, optThroughput);
			++passCount;
		}
	}
}

void PrintPassCounts(const double* passThroughputs, const char* tableName) {
	printf("// %s table generated from bit pass throughputs:\n", tableName);
	for(int i(0); i < MaxBits; ++i)
		printf("// %d bit = %8.3f M\n", i + 1, passThroughputs[i]);
	printf("\nPassTable %s[32] = {\n", tableName);
	for(int targetBits(1); targetBits <= 32; ++targetBits) { 
		int counts[MaxBits];
		int optCounts[MaxBits];
		double optThroughput = 0;
		FindOptimalPath(passThroughputs, counts, MaxBits, targetBits, optCounts,
			optThroughput);

		printf("\t{ %d, %d, %d, %d, %d, %d, %d }%c\t\t// %8.3f M  ( %9.3f M )\n",
			optCounts[0], optCounts[1], optCounts[2], optCounts[3],
			optCounts[4], optCounts[5], optCounts[6],
			(32 == targetBits) ? ' ' : ',', optThroughput, 
			(32.0 / targetBits) * optThroughput);
	}
	printf("};\n\n\n");
}


int main(int argc, char** argv) {
	FILE* f = fopen(argv[1], "r");

	double maxThroughputs[MaxBits] = { 0 };

	while(f && !feof(f)) {
		// Read the table name
		char tableName[128];
		int count = fscanf(f, "%s", tableName);
		if(1 != count) break;

		// Read the six timings.
		double throughputs[MaxBits];
		for(int i(0); i < MaxBits; ++i) {
			int count = fscanf(f, "%lf", &throughputs[i]);
			if(1 != count) return 0;

			maxThroughputs[i] = std::max(maxThroughputs[i], throughputs[i]);
		}
	}

	// Compute the pass table and print it to stdout.
	PrintPassCounts(maxThroughputs, "sort keys");
	return 0;	
}
