#pragma once

#include "../../../util/cucpp.h"
#include "../../../inc/mgpusort.hpp"
#include "../cudpp2/cudpp.h"

// Benchmark parameters for MGPU sort. We expose many options to test all aspects
// of the library.
struct MgpuTerms {
	int numBits;		// Number of bits in the sort.
	int bitPass;		// Target number of bits per pass (MGPU only).
						// If bitPass is zero, use the sortArray API.
	int numThreads;		// Number of threads in the sort block (MGPU only).
	int iterations;	
	bool reset;			// Update from the input array each iteration.
	bool earlyExit;
	bool useTransList;
	int valueCount;		// Set to -1 for index.
	int count;			// Number of elements.

	// Input, read-only, copied with cuMemcpyDtoD into the ping-pong storage.
	CuDeviceMem* randomKeys;
	CuDeviceMem* randomVals[6];

	// Output, copied with cuMemcpyDtoD from the ping-pong storage.
	CuDeviceMem* sortedKeys;
	CuDeviceMem* sortedVals[6];
};

sortStatus_t MgpuBenchmark(MgpuTerms& terms, sortEngine_t engine, double* elapsed);


struct B40cTerms {
	int numBits;
	int iterations;
	bool reset;			// Update from the input array each iteration.
	int count;			// Number of elements.

	CuContext* context;

	// Input, read-only, copied with cuMemcpyDtoD into the ping-pong storage.
	CuDeviceMem* randomKeys;
	CuDeviceMem* randomVals;

	// Output, copied with cuMemcpyDtoD from the ping-pong storage.
	CuDeviceMem* sortedKeys;
	CuDeviceMem* sortedVals;
};

cudaError_t B40cBenchmark(B40cTerms& terms, double* elapsed);


// Use our specially hacked CUDPP and benchmark with B40C's terms.
CUresult CUDPPBenchmark(CUDPPHandle handle, B40cTerms& terms,
	double* elapsed);


// Only support 32-bit keys on key-only sorts.
void ThrustBenchmark(bool reset, int iterations, int count, CuContext* context,
	CuDeviceMem* randomKeys, CuDeviceMem* sortedKeys, double* elapsed);



