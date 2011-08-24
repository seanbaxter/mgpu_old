// MGPU Sort simple test example. Remember to link with the sort library 
// (mgpusort.lib is the import library on Windows).

#include <cstdio>
#include <vector>
#include <random>
#include "../../util/cucpp.h"			// MGPU utility classes
#include "../../inc/mgpusort.hpp"		// MGPU Sort with C++ wrappers

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
	sortStatus_t status = sortCreateEngine("../cubin/", &engine);

	// Create some test data in an std::vector. Generate uints with 32 random
	// bits.
	int NumElements = 1000;
	std::tr1::uniform_int<uint> r(0, 0xffffffff);

	std::vector<uint> hostData(NumElements);
	for(int i(0); i < NumElements; ++i)
		hostData[i] = r(mt19937);

	// Print the unsorted values to screen.
	printf("%d unsorted values:\n", NumElements);
	PrintArrayToScreen(&hostData[0], NumElements);

	// Let MGPU allocate device memory for the source and destination buffers.
	// You can do this yourself with many allocation options. Refer to 
	// mgpusort.h comments for more info. Pass 0 for valueCount to indicate that
	// we just want to sort keys, not key-value tuples.
	sortData_t deviceData;
	status = sortCreateData(engine, NumElements, 0, &deviceData);

	// These are resetting the default values, but it's good to be sure that we
	// are sorting over the desired bits in the key. For best performance only
	// sort over the bits that actually differ.
	deviceData->firstBit = 0;
	deviceData->endBit = 32;

	// Copy the host data to the device array. sortData_d::parity indicates
	// which of the arrays holds the active data. The other array is temp
	// buffer.
	cuMemcpyHtoD(deviceData->keys[deviceData->parity], &hostData[0],
		sizeof(uint) * NumElements);

	// Sort the elements.
	sortArray(engine, deviceData);

	// Retrieve the sorted elements.
	cuMemcpyDtoH(&hostData[0], deviceData->keys[deviceData->parity],
		sizeof(uint) * NumElements);

	// Free the device arrays.
	sortDestroyData(deviceData);

	// Print the unsorted values to screen.
	printf("\n\n%d sorted values:\n", NumElements);
	PrintArrayToScreen(&hostData[0], NumElements);

}

