#include "../../../util/cucpp.h"
#include <vector>
#include <random>

std::tr1::mt19937 mt19937;

typedef unsigned int uint;

#define WARP_SIZE 32

const int NumCountThreads = 128;
const int NumCountWarps = NumCountThreads / WARP_SIZE;

const int NumHistThreads = 1024;

const int NumSortValuesPerThread = 8;

void EasyRadixSort(int numBits, const std::vector<uint>& source,
	std::vector<uint>& dest) {
		
	const int NumDigits = 1<< numBits;
	const int Mask = NumDigits - 1;

	dest.resize(source.size());

	std::vector<uint> counts(NumDigits);
	int count = source.size();
	for(int i(0); i < count; ++i)
		++counts[Mask & source[i]];

	std::vector<uint> scan(NumDigits);
	int prev = 0;
	for(int i(0); i < NumDigits; ++i) {
		scan[i] = prev;
		prev += counts[i];
	}

	for(int i(0); i < count; ++i) {
		uint x = source[i];
		int offset = scan[Mask & x]++;
		dest[offset] = x;
	}
}


bool TestSort(CuContext* context, int numBits, int numBlocks, 
	int numSortThreads) {

	printf("Testing with %d bits.\n", numBits);
		
	const int NumValues = numSortThreads * NumSortValuesPerThread;
	const int NumCountVT = NumValues / WARP_SIZE;

	const int NumDigits = 1<< numBits;
	const int NumChannels = NumDigits / 2;
	const int NumCountBlocks = DivUp(numBlocks, NumCountWarps);
	const int NumElements = NumValues * numBlocks;

	ModulePtr countModule, histModule, sortModule;
	CUresult result = context->LoadModuleFilename(
		"../../src/cubin/count.cubin", &countModule);
	result = context->LoadModuleFilename(
		"../../src/cubin/sorthist.cubin", &histModule);
	result = context->LoadModuleFilename(
		"../../src/cubin/sortloop_128_8_key_simple.cubin", &sortModule);

	char funcName[128];
	sprintf(funcName, "CountBucketsLoop_%d", numBits);
	FunctionPtr countFunc;
	result = countModule->GetFunction(funcName,
		make_int3(NumCountThreads, 1, 1), &countFunc);

	sprintf(funcName, "SortHist_%d", numBits);
	FunctionPtr histFunc;
	result = histModule->GetFunction(funcName,
		make_int3(NumHistThreads, 1, 1), &histFunc);

	sprintf(funcName, "RadixSortLoop_%d", numBits);
	FunctionPtr sortFunc;
	result = sortModule->GetFunction(funcName, 
		make_int3(numSortThreads, 1, 1), &sortFunc);

	int sortBlocksPerSM = sortFunc->BlocksPerSM();
	int countBlocksPerSM = countFunc->BlocksPerSM();
	int numTasks = std::min(numBlocks,
		context->Device()->NumSMs() * sortFunc->BlocksPerSM());

	div_t d = div(numBlocks, numTasks);


	CUtexref texRef;
	sortModule->GetTexRef("keys_texture_in", &texRef);
	result = cuTexRefSetFormat(texRef, CU_AD_FORMAT_UNSIGNED_INT32, 4);
	


	////////////////////////////////////////////////////////////////////////////
	// Generate test data and bucketize in host memory.

	std::tr1::uniform_int<uint> r(0, 0xffffffff);

	std::vector<uint> counts(numBlocks * NumDigits);
	std::vector<uint> digitCounts(NumDigits);

	std::vector<uint> host(NumElements);

	const int Mask = (1<< numBits) - 1;
	for(int i(0); i < NumElements; ++i) {
		host[i] = r(mt19937);
		uint d = Mask & host[i];
		uint index = (i / NumValues) * NumDigits + d;
		++counts[index];
		++digitCounts[d];
	}

	std::vector<uint> digitCountScan(NumDigits);
	int last = 0;
	for(int i(0); i < NumDigits; ++i) {
		digitCountScan[i] = last;
		last += digitCounts[i];
	}


	DeviceMemPtr deviceSource, deviceScan, deviceTotals;
	result = context->MemAlloc(host, &deviceSource);
	result = context->MemAlloc<uint>(numBlocks * NumChannels, &deviceScan);
	result = context->MemAlloc<uint>(numTasks * NumDigits, &deviceTotals);


	////////////////////////////////////////////////////////////////////////////
	// Run the count kernel and get counts in device memory.

	CUdeviceptr nullPtr = 0;
	CuCallStack callStack;
	callStack.Push(deviceSource, 0, NumCountVT, d.quot, d.rem, deviceScan, 
		deviceTotals);

	result = countFunc->Launch(numTasks, 1, callStack);

	std::vector<uint> hostScan;
	result = deviceScan->ToHost(hostScan);

	std::vector<uint> hostTotals;
	result = deviceTotals->ToHost(hostTotals);

	std::vector<uint> blockScanHost(numBlocks * NumDigits);

	for(int block(0); block < numBlocks; ++block)
		for(int d(0); d < NumDigits; ++d) {

			uint index = (NumDigits / 2 - 1) & d;
			uint packed = hostScan[block * NumDigits / 2 + index];
			packed /= 4;

			if(d >= NumDigits / 2) packed>>= 16;
			packed &= 0xffff;
			uint next = NumValues;
			if(d < NumDigits - 1) {
				uint d2 = d + 1;
				uint index = (NumDigits / 2 - 1) & d2;
				uint packed2 = hostScan[block * NumDigits / 2 + index];
				packed2 /= 4;
				if(d2 >= NumDigits / 2) packed2>>= 16;
				packed2 &= 0xffff;
				next = packed2;
			}

			uint count = next - packed;

			blockScanHost[block * NumDigits + d] = packed;

			index = block * NumDigits + d;
			if(count != counts[index]) {
				printf("Error on block %d digit %d. Has %d wants %d\n", block,
					d, count, counts[index]);
				exit(0);
			}
		}

	printf("Count launch correct.\n");


	////////////////////////////////////////////////////////////////////////////
	// Run a histogram pass.

	DeviceMemPtr totalsScanDevice;
	result = context->MemAlloc<uint>(NumDigits, &totalsScanDevice);

	callStack.Reset();
	callStack.Push(deviceTotals, numTasks, totalsScanDevice);
	result = histFunc->Launch(1, 1, callStack);

	std::vector<uint> taskOffsetsHost;
	deviceTotals->ToHost(taskOffsetsHost);

	std::vector<uint> totalsScanHost;
	totalsScanDevice->ToHost(totalsScanHost);

	// Construct scatter offsets for each block.
	std::vector<uint> blockOffsets(numBlocks * NumDigits);
	std::vector<uint> taskOffsets2 = taskOffsetsHost;
	for(int task(0); task < numTasks; ++task) {
		int x = d.quot * task;
		x += std::min(task, d.rem);
		int y = x + d.quot + (task < d.rem);

		// Get the per-digit count warp offsets.
		uint* countOffsets = &taskOffsets2[0] + task * NumDigits;

		for(int i(x); i < y; ++i) {
			// Load the counts for each digit.
			for(int d(0); d < NumDigits; ++d) {
				uint digitCount = counts[i * NumDigits + d];
				blockOffsets[i * NumDigits + d] = countOffsets[d];
				countOffsets[d] += digitCount;
			}
		}
	}

	// Compute scatter offsets from host counts.
	std::vector<uint> blockOffsetsHost(numBlocks * NumDigits);
	last = 0;
	for(int d(0); d < NumDigits; ++d)
		for(int block(0); block < numBlocks; ++block) {
			uint index = block * NumDigits + d;
			blockOffsetsHost[index] = last;
			last += counts[index];

			if(blockOffsetsHost[index] != blockOffsets[index]) {
				printf("Error on block %d digit %d.\n", block, d);
			}
		}


	////////////////////////////////////////////////////////////////////////////
	// 

	DeviceMemPtr sortedDevice;
	result = context->MemAlloc<uint>(NumElements, &sortedDevice);


	size_t offset;
	result = cuTexRefSetAddress(&offset, texRef, deviceSource->Handle(),
		4 * NumElements);

	callStack.Reset();
	callStack.Push(deviceSource, deviceScan, deviceTotals, 0, sortedDevice,
		d.quot, d.rem, (CUdeviceptr)0);
					
	result = sortFunc->Launch(numTasks, 1, callStack);

	std::vector<uint> sortedHost;
	result = sortedDevice->ToHost(sortedHost);

	std::vector<uint> sortedHost2;
	EasyRadixSort(numBits, host, sortedHost2);
	
	for(int i(0); i < NumElements; ++i) {
		if(sortedHost2[i] != sortedHost[i]) {
			printf("%d\n", i);
			exit(0);
		}
	}

	return 0;	
}

int main(int argc,  char** argv) {

	cuInit(0);

	DevicePtr device;
	CUresult result = CreateCuDevice(0, &device);
	
	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	for(int numBits = 1; numBits <= 7; ++numBits) {
		TestSort(context, numBits, 101, 128);
	}

}
