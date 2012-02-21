#include "../../../util/cucpp.h"
#include <vector>
#include <random>

std::tr1::mt19937 mt19937;

typedef unsigned int uint;

#define WARP_SIZE 32

const int NumCountThreads = 128;
const int NumCountWarps = NumCountThreads / WARP_SIZE;

const int NumHistThreads = 1024;

const int NumDownsweepThreads = 128;

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
		uint x = source[i];int offset = scan[Mask & x]++;
		dest[offset] = x;
	}
}



bool TestSort(CuContext* context, int numBits, int numBlocks, 
	int numSortThreads, int valuesPerThread) {

	const int NumValues = numSortThreads * valuesPerThread;
	const int NumDigits = 1<< numBits;
	const int NumChannels = NumDigits / 2;

	int numElements = NumValues * numBlocks;

	printf("Loading %d kernels.\n", numBits);

	ModulePtr countModule, histModule, downsweepModule, sortModule;
	FunctionPtr countFunc, histFunc, downsweepFunc, sortFunc;

	CUresult result = context->LoadModuleFilename(
		"../../src/cubin/count.cubin", &countModule);
	
	result = context->LoadModuleFilename(
		"../../src/cubin/sorthist.cubin", &histModule);

	result = context->LoadModuleFilename(
		"../../src/cubin/sort2.cubin", &sortModule);

	result = context->LoadModuleFilename(
		"../../src/cubin/sortdownsweep.cubin", &downsweepModule);

	char funcName[128];
	sprintf(funcName, "CountBucketsLoop_%d", numBits);
	result = countModule->GetFunction(funcName, 
		make_int3(NumCountThreads, 1, 1), &countFunc);

	sprintf(funcName, "SortHist_%d", numBits);
	result = histModule->GetFunction(funcName,
		make_int3(1024, 1, 1), &histFunc);

	sprintf(funcName, "SortDownsweep_%d", numBits);
	result = downsweepModule->GetFunction(funcName, 
		make_int3(NumDownsweepThreads, 1, 1), &downsweepFunc);

	sprintf(funcName, "RadixSort_%d_%d_%d", numBits, valuesPerThread,
		numSortThreads);
	result = sortModule->GetFunction(funcName,
		make_int3(numSortThreads, 1, 1), &sortFunc);

	CUtexref keys_texture_in;
	result = sortModule->GetTexRef("keys_texture_in", &keys_texture_in);
	result = cuTexRefSetFormat(keys_texture_in, CU_AD_FORMAT_UNSIGNED_INT32, 4);


	////////////////////////////////////////////////////////////////////////////
	// Generate digit arrays.

	std::tr1::uniform_int<uint> r(0, NumDigits - 1);

	std::vector<uint> counts(NumDigits * numBlocks);
	std::vector<uint> host(NumValues * numBlocks);

	for(int b(0); b < numBlocks; ++b) {
		uint index = b * NumValues;
		for(int i(index); i < index + NumValues; ++i) {
			host[i] = r(mt19937);
			++counts[b * NumDigits + host[i]];
		}
	}

	
	std::vector<uint> digitScan(NumDigits);
	int last = 0;

	for(int i(0); i < NumDigits; ++i) {
		digitScan[i] = last;
		for(int b(0); b < numBlocks; ++b) {
			last += counts[i + b * NumDigits];
		}
	}

	DeviceMemPtr deviceSource;
	result = context->MemAlloc(host, &deviceSource);

	size_t offset;
	result = cuTexRefSetAddress(&offset, keys_texture_in, deviceSource->Handle(),
		4 * NumValues);


	////////////////////////////////////////////////////////////////////////////
	// Generate counts.

	printf("Generating %d counts.\n", numBits);

	int numSMs = context->Device()->NumSMs();
	int numTasks = std::min(countFunc->BlocksPerSM() * numSMs, numBlocks);
	int numCountBlocks = DivUp(numTasks, NumCountThreads / WarpSize);

	div_t d = div(numBlocks, numTasks);

	DeviceMemPtr deviceCounts, deviceTaskOffsets;
	result = context->MemAlloc<uint>(numBlocks * NumChannels, &deviceCounts);
	result = context->MemAlloc<uint>(numTasks * NumDigits, &deviceTaskOffsets);

	CuCallStack callStack;
	callStack.Push(deviceSource, 0, NumValues / WARP_SIZE, d.quot, d.rem,
		numBlocks, deviceCounts, deviceTaskOffsets);
	result = countFunc->Launch(numCountBlocks, 1, callStack);

	std::vector<uint> countsHost, taskOffsetsHost;
	deviceCounts->ToHost(countsHost);
	deviceTaskOffsets->ToHost(taskOffsetsHost);


	////////////////////////////////////////////////////////////////////////////
	// Run the histogram pass.

	printf("Generating %d hist.\n", numBits);

	
	DeviceMemPtr totalScan;
	result = context->MemAlloc<uint>(NumDigits, &totalScan);

	callStack.Reset();
	callStack.Push(deviceTaskOffsets, numTasks, totalScan);
	histFunc->Launch(1, 1, callStack);

	std::vector<uint> taskOffsetsHost2;
	deviceTaskOffsets->ToHost(taskOffsetsHost2);

	std::vector<uint> totalScanHost;
	totalScan->ToHost(totalScanHost);

	////////////////////////////////////////////////////////////////////////////
	// Run the downsweep pass.

	printf("Generating %d downsweep.\n", numBits);

	DeviceMemPtr deviceGlobalScan;
	result = context->MemAlloc<uint>(numBlocks * NumDigits, &deviceGlobalScan);

	const int ColumnWidth = std::min(WarpSize, NumDigits);
	const int NumColumns = NumDownsweepThreads / ColumnWidth;
	const int numDownsweepBlocks = DivUp(numTasks, NumColumns);

	callStack.Reset();
	callStack.Push(deviceTaskOffsets, d.quot, d.rem, numBlocks, NumValues, 
		deviceCounts, deviceGlobalScan);
	downsweepFunc->Launch(numDownsweepBlocks, 1, callStack);

	std::vector<uint> globalScanHost;
	deviceGlobalScan->ToHost(globalScanHost);


	////////////////////////////////////////////////////////////////////////////
	// Run the sort pass

	printf("Generating %d sort.\n", numBits);

	DeviceMemPtr sortedDevice;
	result = context->MemAlloc<uint>(numElements, &sortedDevice);
	
	result = cuTexRefSetAddress(&offset, keys_texture_in,
		deviceSource->Handle(), 4 * numElements);

	callStack.Reset();
	callStack.Push(deviceSource, deviceGlobalScan, 0, sortedDevice);

	result = sortFunc->Launch(numBlocks, 1, callStack);

	std::vector<uint> sortedHost;
	sortedDevice->ToHost(sortedHost);






	int i = 0;

/*





	sortFunc->Launch(1, 1, callStack);

	std::vector<uint> scatter;
	deviceScatter->ToHost(scatter);

	std::vector<uint> counts2(NumDigits);
	for(int i(0); i < numSortThreads; ++i) {
		const uint* packed = &scatter[i * (NumDigits / 2)];
		for(int d(0); d < NumDigits / 2; ++d) {
			uint x = packed[d];
			counts2[d] += 0xffff & x;
			counts2[d + NumDigits / 2] += x>> 16;
		}
	}




	int i = 0;

	return 0;
	/*

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
*/
	return 0;	
}

int main(int argc,  char** argv) {
/*
	int valuesPerThread = 24;
	int numValues = valuesPerThread * WarpSize;
	std::vector<uint> shared(numValues + numValues / 8);

	std::vector<uint> banks(valuesPerThread);
	
	for(int i(0); i < numValues; ++i) {
		shared[i + i / 8] = i;
	}

	for(int tid(0); tid < 32; ++tid) {
		printf("tid = %2d:\n", tid);

		int index = valuesPerThread * tid;
		index += index / 8;

		for(int v(0); v < 8; ++v) {
			int x = shared[index];
			int bank = index % 32;
			banks[v] |= 1<< bank;
			printf("\t%3d (%2d)\n", x, bank);
			++index;
		}

		++index;
		for(int v(0); v < 8; ++v) {
			int x = shared[index];
			int bank = index % 32;
			banks[8 + v] |= 1<< bank;
			printf("\t%3d (%2d)\n", x, bank);
			++index;
		}

		++index;
		for(int v(0); v < 8; ++v) {
			int x = shared[index];
			int bank = index % 32;
			banks[16 + v] |= 1<< bank;
			printf("\t%3d (%2d)\n", x, bank);
			++index;
		}

	}
	for(int tid(0); tid < 32; ++tid) {
		int index = valuesPerThread * tid;
		int index2 = index + index / 8;
		int bank = index2 % 32;
		printf("%2d: %3d %3d %2d\n", tid, index, index2, bank);		
	}
	return 0;
*/






	for(int i(0); i < 35; ++i) {
		printf("%d %d\n", i, RoundUp(i, 13));

	}







	cuInit(0);

	DevicePtr device;
	CUresult result = CreateCuDevice(0, &device);
	
	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	for(int numBits = 1; numBits <= 7; ++numBits) {
		TestSort(context, numBits, 150, 64, 24);
	}

}
