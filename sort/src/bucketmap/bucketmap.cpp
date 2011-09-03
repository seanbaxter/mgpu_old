#define _USE_MATH_DEFINES

#define NOMINMAX
#include <windows.h>  
  
#include <iostream>
#include <cstdio>
#include <utility>
#include <algorithm>
#include <vector>

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

typedef unsigned int uint;
typedef std::pair<int, int> IntPair;
typedef std::pair<uint, uint> UintPair;

const uint WARP_SIZE = 32;

// random number generator - make global to prevent re-initialization.
std::tr1::mt19937 mt19937;

uint NumTransactions(uint histOffset, uint keyCount) {
	uint start = ~(WARP_SIZE - 1) & histOffset;
	uint end = ~(WARP_SIZE - 1) & (histOffset + keyCount + WARP_SIZE - 1);
	uint count = (end - start) / WARP_SIZE;
	return keyCount ? count : 0;
}
UintPair RequestInterval(uint histOffset, uint keyCount, uint r) {
	uint start = ~(WARP_SIZE - 1) & (histOffset + r * WARP_SIZE);
	uint end = std::min(start + WARP_SIZE, histOffset + keyCount);
	start = std::max(start, histOffset);
	return UintPair(start, end);
}
 
typedef std::pair<int, int> Pair;

Pair CountTransactions(const int* keys, const int* gather, const int* scatter, 
	int numBuckets, int numValues) {

	int simpleTransCount = 0;

	std::vector<Pair> intervals;

	for(int warp(0); warp < numValues / (int)WARP_SIZE; ++warp) {
		const int* k = keys + warp * WARP_SIZE;
		
		for(int i(0); i < (int)WARP_SIZE; ++i) {
			int index = warp * WARP_SIZE + i;
			int key = k[i];
			if(!i || intervals.back().first != key)
				intervals.push_back(Pair(key, index));
		}

		// Get the simple transaction count
		for(int i(0); i < (int)intervals.size(); ++i) {
			Pair interval = intervals[i];
			int end = (i < (int)intervals.size() - 1) ? 
				intervals[i + 1].second : 
				((warp + 1) * WARP_SIZE);
			int count = end - interval.second;

			// get the scatter offset
			int index = interval.second;
			int key = keys[index];
			int scatterOffset = scatter[key] + index - gather[key];

			simpleTransCount += NumTransactions(scatterOffset, count);
		}
		intervals.clear();
	}

	int optTransCount = 0;

	for(int key(0); key < numBuckets; ++key) {
		// get the interval and count for each key
		int first = gather[key];
		int end = (key < numBuckets - 1) ? gather[key + 1] : numValues;
		int count = end - first;
		if(count) 
			optTransCount += NumTransactions(scatter[key], count);
	}

	return Pair(optTransCount, simpleTransCount);
}

Pair ComputeTransactionCounts(int numBits, int numValues) {
	int numBuckets = 1<< numBits;


	// generate the keys and bucket counts
	std::tr1::uniform_int<> r(0, numBuckets - 1), r2(0, WARP_SIZE - 1);
	std::vector<int> keys(numValues);
	
	std::vector<int> bucketCounts(numBuckets);
	for(int i(0); i < numValues; ++i) {
		keys[i] = r(mt19937);
		++bucketCounts[keys[i]];
	}
	std::sort(keys.begin(), keys.end());

	// generate the scatter offsets and create the gather offsets
	std::vector<int> scatter(numBuckets), gather(numBuckets);
	int lastGather = 0;
	for(int i(0); i < numBuckets; ++i) {
		gather[i] = lastGather;
		lastGather += bucketCounts[i];

		scatter[i] = r2(mt19937);
	}

	Pair transCountPair = CountTransactions(&keys[0], &gather[0], &scatter[0],
		numBuckets, numValues);
	return transCountPair;
}


int main() {

	const int FirstTest = 512;
	const int NumTests = 3;				// 512, 1024, 2048
	const int MaxBits = 7;				// 1 - 7 inclusive
	const int NumIterations = 100;

	printf("Calculating average transaction behavior for MGPU Sort library.\n");
	printf("Each test run over %d iterations.\n\n", NumIterations);

	// Display with vals first, then numBits, then block size.
	for(int vals(0); vals <= 6; ++vals) {
		printf("VALUE COUNT=%d\n", vals);

		for(int numBits(1); numBits <= MaxBits; ++numBits) {
			printf(" BITS=%d\n", numBits);

			for(int test(0); test < NumTests; ++test) {
				int numValues = FirstTest<< test;
				int numWarps = numValues / WARP_SIZE;
				printf("  VALS=%4d  ", numValues);

				double opt = 0, simple = 0;
				for(int it(0); it < NumIterations; ++it) {
					Pair pair = ComputeTransactionCounts(numBits, numValues);
					opt += pair.first;
					simple += pair.second;
				}
				opt /= NumIterations * numWarps;
				simple /= NumIterations * numWarps;

				double optWrite = (1 + vals) * opt / numBits;
				double optTotal = ((1 + vals) * (1 + opt) + 1) / numBits;
				printf("opt=%4.2f/%4.2f/%4.2f  ", opt, optWrite, optTotal);

				double simWrite = (1 + vals) * simple / numBits;
				double simTotal = ((1 + vals) * (1 + simple) + 1) / numBits;
				printf("sim=%4.2f/%5.2f/%5.2f  ", simple, simWrite, simTotal);

				double ratioWrite = 100 * optWrite / simWrite;
				double ratioTotal = 100 * optTotal / simTotal;
				printf("ratio=%4.1f%%/%4.1f%%\n", ratioWrite, ratioTotal);
			}
		}
	}
}


