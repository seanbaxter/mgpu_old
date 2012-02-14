// MGPU Sort simple test 2 example. Remember to link with the sort library 
// (mgpusort.lib is the import library on Windows).
#include <cstdio>
#include <vector>
#include "../../../util/cucpp.h"			// MGPU utility classes
#include "../../../inc/mgpusort.hpp"		// MGPU Sort with C++ wrappers

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

// Mersenne twister RNG
std::tr1::mt19937 mt19937;

void PrintArrayToScreen(const uint* values, int numValues) {
	while(numValues > 0) {
		int numValsThisLine = std::min(8, numValues);
		for(int i(0); i < numValsThisLine; ++i)
			printf("%08x ", values[i]);
		values += numValsThisLine;
		numValues -= numValsThisLine;
		printf("\n");
	}
}

void PrintArrayTuples(const uint* keys, const uint* val1, const float* val2,
	int count) {

	for(int i(0); i < count; ++i) {
		printf("%08x ", keys[i]);
		if(val1) printf("%10d ", val1[i]);
		if(val2) printf("%5.3f", val2[i]);
		printf("\n");
	}
}


int main(int argc, char** argv) {

	// Initialize the CUDA driver API library.
	CUresult result = cuInit(0);

	// Create a CUDA device and context.
	DevicePtr device;
	result = CreateCuDevice(0, &device);

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	// Initialize the MGPU Sort engine. This is the C interface, but you can 
	// also use the RAII MgpuSort type in mgpusort.hpp. Pass the directory name
	// that holds the .cubin kernel files. If these aren't already built, use
	// build_all.bat in /kernels.
	sortEngine_t engine;
	sortStatus_t status = sortCreateEngine("../../src/cubin/", &engine);

	// Create some test data in an std::vector. Generate uits with 18 set random
	// bits, starting at bit 7.
	int NumElements = 128;
	std::tr1::uniform_int<uint> r(0, (1<< 18) - 1);

	std::vector<uint> hostKeys(NumElements);
	std::vector<uint> hostValues1(NumElements);
	std::vector<float> hostValues2(NumElements);
	for(int i(0); i < NumElements; ++i) {
		hostKeys[i] = r(mt19937)<< 7;
		hostValues1[i] = i;			// Cache the original index of the data.
		hostValues2[i] = sqrtf((float)i);	// Cache a second value.		
	}

	// Print the unsorted values to screen.
	printf("%d unsorted values:\n", NumElements);
	PrintArrayToScreen(&hostKeys[0], NumElements);

	// Copy the key and value arrays into device memory. Note that the MGPU Sort
	// library needs buffers that are rounded up in size to a multiple of 2048.
	// This is an inconvenience but simplifies execution of the kernels.
	int RoundedSize = RoundUp(NumElements, 2048);
	DeviceMemPtr deviceKeys, deviceValues1, deviceValues2;
	context->MemAlloc<uint>(RoundedSize, &deviceKeys);
	context->MemAlloc<uint>(RoundedSize, &deviceValues1);
	context->MemAlloc<float>(RoundedSize, &deviceValues2);

	// Fill the buffers with source data.
	deviceKeys->FromHost(hostKeys);
	deviceValues1->FromHost(hostValues1);
	deviceValues2->FromHost(hostValues2);

	// Use MgpuSortData from mgpusort.hpp to help manage using client-allocated
	// memory alongside library-allocated memory. Allocate enough space to hold
	// our keys and values (2).
	MgpuSortData sortData;
	sortData.AttachKey(deviceKeys->Handle());
	sortData.AttachVal(0, deviceValues1->Handle());
	sortData.AttachVal(1, deviceValues2->Handle());

	// MgpuSortData::Alloc calls sortAllocData. This allocates the temporary
	// buffers required for ping-pong operation.
	sortData.Alloc(engine, NumElements, 2);

	// Set the range of the key bits to sort.
	sortData.firstBit = 7;
	sortData.endBit = 7 + 18;

	// Sort the elements. This internally ping-pongs the buffers. Buffers [0]
	// always hold the active data, and buffers [1] are always considered
	// temporary. The key and values device arrays we attached are now in
	// sortData.keys[sortData.parity] and values1[parity], values2[parity].
	sortArray(engine, &sortData);

	// Read the sorted data back into host memory.
	std::vector<uint> sortedKeys(NumElements);
	std::vector<uint> sortedValues1(NumElements);
	std::vector<float> sortedValues2(NumElements);
	cuMemcpyDtoH(&sortedKeys[0], sortData.keys[0], 4 * NumElements);
	cuMemcpyDtoH(&sortedValues1[0], sortData.values1[0], 4 * NumElements);
	cuMemcpyDtoH(&sortedValues2[0], sortData.values2[0], 4 * NumElements);

	PrintArrayTuples(&sortedKeys[0], &sortedValues1[0], &sortedValues2[0],
		NumElements);

	// Reset the MgpuSortData struct. This frees only those device arrays that
	// were allocated by the library.
	sortData.Reset();

	// Copy the unsorted keys back into deviceKeys and prepare for an index 
	// sort. -1 for valueCount generates indices as the first value stream.
	deviceKeys->FromHost(hostKeys);
	sortData.AttachKey(deviceKeys->Handle());
	sortData.Alloc(engine, NumElements, -1);

	// This time set over the fill 32-bit range and enable early exit. It will
	// skip the first and last pass of the sort, as only the middle bits of the
	// key are randomized.
	sortData.firstBit = 0;
	sortData.endBit = 32;
	sortArray(engine, &sortData);

	// Print the sorted keys and indices (values1). These should match the first
	// value stream sorted in the earlier pass.
	result = cuMemcpyDtoH(&sortedKeys[0], sortData.keys[0], 4 * NumElements);
	result = cuMemcpyDtoH(&sortedValues1[0], sortData.values1[0], 
		4 * NumElements);

	// The sortData.values1[0] array was generated by the library to refer
	// to the indices of each key in the source sequence.
	PrintArrayTuples(&sortedKeys[0], &sortedValues1[0], 0, NumElements);
	
	// Free the sort buffers.
	sortData.Reset();

	// Free the sort engine.
	sortReleaseEngine(engine);
}
