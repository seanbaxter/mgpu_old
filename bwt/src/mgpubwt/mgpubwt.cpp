#include "../../../inc/mgpubwt.h"
#include "../../../inc/mgpusort.hpp"
#include "../../../util/cucpp.h"
#include <memory>
#include <sstream>
#include <algorithm>

typedef unsigned char byte;

const int BWTGatherThreads = 512;
const int BWTCompareThreads = 512;
const int TinyBlockCutoff = 3000;

// Returns a textual representation of the enums.
const char* BWTAPI bwtStatusString(bwtStatus_t status);

const char* BwtStatusStrings[] = {
	"BWT_STATUS_SUCCESS",
	"BWT_STATUS_NOT_INITIALIZED",
	"BWT_STATUS_DEVICE_ALLOC_FAILED",
	"BWT_STATUS_INVALID_CONTEXT",
	"BWT_STATUS_KERNEL_NOT_FOUND",
	"BWT_STATUS_KERNEL_ERROR",
	"BWT_STATUS_LAUNCH_ERROR",
	"BWT_STATUS_INVALID_VALUE",
	"BWT_STATUS_DEVICE_ERROR",
	"BWT_STATUS_NOT_IMPLEMENTED",
	"BWT_STATUS_UNSUPPORTED_DEVICE",
	"BWT_STATUS_SORT_ERROR"
};

const char* BWTAPI bwtStatusString(bwtStatus_t status) {
	if(status > sizeof(BwtStatusStrings) / sizeof(char*)) return 0;
	return BwtStatusStrings[status];
}


////////////////////////////////////////////////////////////////////////////////
// MGPU BWT FUNCTIONS

// Create the bwt engine on the CUDA device API context
struct bwtEngine_d {
	ContextPtr context;
	DeviceMemPtr sortSpace, bwtSpace;
	CUtexref gatherTexRef;
	
	ModulePtr module;
	FunctionPtr gatherKeys[6];
	FunctionPtr compareKeys[6];

	std::vector<byte> symbols;
	std::vector<byte> indices;
	std::vector<byte> flags;
	
	sortEngine_t sortEngine;

	bwtEngine_d() { 
		sortEngine = 0; 
	}
	~bwtEngine_d() {
		sortReleaseEngine(sortEngine);
	}
};

struct BlockPointers {
	CUdeviceptr keys, headFlags;
	int totalSize;
	int flagsOffset;
};


////////////////////////////////////////////////////////////////////////////////
// bwcCreateEngine, bwtDestroyEngine

// Create the engine and attach to the current context. The kernelPath must be
// a directory containing both the MGPU Sort .cubins and bwt.cubin.
bwtStatus_t BWTAPI bwtCreateEngine(const char* kernelPath, 
	bwtEngine_t* engine) {

	std::auto_ptr<bwtEngine_d> e(new bwtEngine_d);
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result || 
		2 != e->context->Device()->ComputeCapability().first)
		return BWT_STATUS_UNSUPPORTED_DEVICE;

	// Load the BWT module and functions.
	result = e->context->LoadModuleFilename(
		std::string(kernelPath) + "bwt.cubin", &e->module);
	if(CUDA_SUCCESS != result) return BWT_STATUS_KERNEL_NOT_FOUND;

	result = cuModuleGetTexRef(&e->gatherTexRef, e->module->Handle(),
		"string_texture");
	if(CUDA_SUCCESS != result) return BWT_STATUS_KERNEL_ERROR;

	cuTexRefSetFlags(e->gatherTexRef, CU_TRSF_READ_AS_INTEGER);
	cuTexRefSetFormat(e->gatherTexRef, CU_AD_FORMAT_UNSIGNED_INT32, 1);

	for(int i(0); i < 6; ++i) {
		std::ostringstream oss;
		oss<< "GatherBWTKeys_"<< (i + 1);
		result = e->module->GetFunction(oss.str(), 
			make_int3(BWTGatherThreads, 1, 1), &e->gatherKeys[i]);
		if(CUDA_SUCCESS != result) return BWT_STATUS_KERNEL_ERROR;

		oss.str("");
		oss<< "CompareBWTKeys_"<< (i + 1);
		result = e->module->GetFunction(oss.str(),
			make_int3(BWTCompareThreads, 1, 1), &e->compareKeys[i]);
		if(CUDA_SUCCESS != result) return BWT_STATUS_KERNEL_ERROR;
	}

	// Load the sort engine.
	sortStatus_t sortStatus = sortCreateEngine(kernelPath, &e->sortEngine);
	if(SORT_STATUS_SUCCESS != sortStatus) return BWT_STATUS_SORT_ERROR;

	*engine = e.release();
	return BWT_STATUS_SUCCESS;
}

bwtStatus_t BWTAPI bwtDestroyEngine(bwtEngine_t engine) {
	if(engine) delete engine;
	return BWT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Device memory allocation for bwtSortBlock. Allocated memory is kept in the
// engine object so that it needn't be re-allocated every call. CUDA uses a
// driver API so allocations are expensive.

CUresult AllocSortMem(bwtEngine_d* engine, int count, int numStreams,
	CUdeviceptr streams[7][2]) {

	// Round up to a multiple of 2048 to satisfy MGPU Sort.
	size_t count2 = RoundUp(count, 2048);

	// Each element is 4 bytes.
	size_t size = 4 * count2;

	// Allocate two pairs of streams for ping-pong in the sort for each DWORD of
	// keys plus a pair for the indices, plus another array to hold the original
	// keys.
	size_t totalSize = (numStreams + 1) * 2 * size;

	if(!engine->sortSpace.get() || (engine->sortSpace->Size() < totalSize)) {
		engine->sortSpace.reset();
		CUresult result = engine->context->ByteAlloc(totalSize,
			&engine->sortSpace);
		if(CUDA_SUCCESS != result) return result;
	}

	// Assign the memory to the symbol streams.
	int offset = 0;
	for(int i(0); i < numStreams; ++i) {
		streams[i][0] = engine->sortSpace->Handle() + offset;
		streams[i][1] = streams[i][0] + size;
		offset += 2 * size;		
	}

	// Assign memory to the indices.
	streams[6][0] = engine->sortSpace->Handle() + offset;
	streams[6][1] = streams[6][0] + size;

	return CUDA_SUCCESS;
}

CUresult AllocBWTMem(bwtEngine_d* engine, int count, BlockPointers& pointers) {
	size_t keySize = 4 * RoundUp(count + 32, 2048);
	size_t flagSize = RoundUp(count, 2048);
	size_t totalSize = keySize + flagSize;

	if(!engine->bwtSpace.get() || (engine->bwtSpace->Size() < totalSize)) {
		engine->bwtSpace.reset();
		CUresult result = engine->context->ByteAlloc(totalSize, 
			&engine->bwtSpace);
		if(CUDA_SUCCESS != result) return result;
	}
	pointers.keys = engine->bwtSpace->Handle();
	pointers.headFlags = pointers.keys + keySize;
	pointers.totalSize = totalSize;
	pointers.flagsOffset = keySize;

	return CUDA_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// HostSortRange

// On the CPU uses a comparison sort (std::sort) to sort elements that fall 
// within the same segment (that is, their first gpuKeySize elements are the
// same). This predicate object does a truncated string compare between indices
// a and b.

struct QSortPred {
	const byte* symbols;
	int count2;
	int gpuKeySize;

	bool operator()(int a, int b) const {
		int result = memcmp(symbols + a + gpuKeySize, symbols + b + gpuKeySize,
			count2);
		return (result < 0) || (!result && (a < b));
	}
};


////////////////////////////////////////////////////////////////////////////////
// TinyBlockSort
// If the string is very small, sort directly on CPU as GPU utilization will be
// poor.

void TinyBlockSort(const void* block, int count, void* transform, int* indices,
	int* segCount, float* avSegSize) {

	std::vector<byte> symbols(2 * count);
	memcpy(&symbols[0], block, count);
	memcpy(&symbols[0] + count, block, count);

	std::vector<int> hostIndices(count);
	for(int i(0); i < count; ++i)
		hostIndices[i] = i;

	QSortPred sortPred;
	sortPred.symbols = &symbols[0];
	sortPred.count2 = count;
	sortPred.gpuKeySize = 0;

	std::sort(&hostIndices[0], &hostIndices[0] + count, sortPred);

	if(segCount) *segCount = 1;
	if(avSegSize) *avSegSize = (float)count;

	if(indices) memcpy(indices, &hostIndices[0], 4 * count);
	
	if(transform) {
		byte* xform = (byte*)transform;
		for(int i(0); i < count; ++i) {
			int index = hostIndices[i] + count - 1;
			xform[i] = symbols[index];
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// bwtSortBlock

bwtStatus_t BWTAPI bwtSortBlock(bwtEngine_t engine, const void* block, 
	int count, int gpuKeySize, void* transform, int* indices, int* segCount, 
	float* avSegSize) {

	// Handle very small arrays entirely in CPU. No point involving the GPU.
	if(count < TinyBlockCutoff) {
		TinyBlockSort(block, count, transform, indices, segCount, avSegSize);
		return BWT_STATUS_SUCCESS;
	}

	// Check for the arguments.
	if(!engine || gpuKeySize > 24) return BWT_STATUS_INVALID_VALUE;
	int numStreams = DivUp(gpuKeySize, 4);

	////////////////////////////////////////////////////////////////////////////
	// Allocate space and generate the keys and indices.

	// Copy the symbols twice into a temporary array. This makes the string
	// comparisons easier as we don't have to make out-of-bounds checks.
	engine->symbols.resize(2 * count);
	memcpy(&engine->symbols[0], block, count);
	memcpy(&engine->symbols[0] + count, block, count);

	// Allocate device space for key generation and sorting.
	BlockPointers pointers;
	CUresult result = AllocBWTMem(engine, count, pointers);
	if(CUDA_SUCCESS != result) return BWT_STATUS_DEVICE_ALLOC_FAILED;

	CUdeviceptr streams[7][2];
	result = AllocSortMem(engine, count, numStreams, streams);

	// Copy the symbols to device memory and gather the keys.
	cuMemcpyHtoD(pointers.keys, &engine->symbols[0], count + 32);

	CuCallStack callStack;
	callStack.Push(pointers.keys, count, streams[0][0], streams[1][0],
		streams[2][0], streams[3][0], streams[4][0], streams[5][0],
		streams[6][0]);
	result = engine->gatherKeys[numStreams - 1]->Launch(
		DivUp(count, BWTGatherThreads), 1, callStack);
	if(CUDA_SUCCESS != result) return BWT_STATUS_LAUNCH_ERROR;


	////////////////////////////////////////////////////////////////////////////
	// Perform the GPU radix sort over gpuKeySize bytes.

	sortData_d sortData;
	sortData.maxElements = count;
	sortData.numElements = count;
	sortData.preserveEndKeys = false;
	sortData.earlyExit = false;
	sortData.parity = 0;

	// Sort each stream of keys from least sig to most sig and jettison them
	// as they are sorted. Keep the most-sig keys and the indices around.
	for(int curStream(numStreams - 1); curStream >= 0; --curStream) {
		int p = sortData.parity;

		// Set the key to sort.
		sortData.keys[0] = streams[curStream][p];
		sortData.keys[1] = streams[curStream][1 - p];
		
		for(int i(0); i < curStream; ++i) {
			// Set the keys we have yet to sort.
			sortData.values[i][0] = streams[i][p];
			sortData.values[i][1] = streams[i][1 - p];
		}

		// Set the indices.
		sortData.values[curStream][0] = streams[6][p];
		sortData.values[curStream][1] = streams[6][1 - p];

		sortData.valueCount = curStream + 1;
		int curBytes = std::min(gpuKeySize - 4 * curStream, 4);
		sortData.firstBit = 32 - 8 * curBytes;
		sortData.endBit = 32;
		
		sortStatus_t sortStatus = sortArray(engine->sortEngine, &sortData);
		if(SORT_STATUS_SUCCESS != sortStatus)
			return BWT_STATUS_SORT_ERROR;
	}


	////////////////////////////////////////////////////////////////////////////
	// Compare the first gpuKeySize bytes of adjacent elements in the sorted
	// array. Every element that differs from its immediate predecessor starts
	// its own segments. Segments with two or more elements need to be sorted
	// with a comparison sort (quicksort) on the CPU.

	uint lastMask = 0xffffffff;
	if(3 & gpuKeySize) lastMask<<= 8 * (4 - (3 & gpuKeySize));

	// Call CompareBWTKeys. Note that the sorted indices are in values1[0].
	size_t texByteOffset;
	result = cuTexRefSetAddress(&texByteOffset, engine->gatherTexRef,
		pointers.keys, count + 32);

	callStack.Reset();
	callStack.Push(sortData.values1[0], count, lastMask, pointers.headFlags);
	result = engine->compareKeys[numStreams - 1]->Launch(
		DivUp(count, BWTCompareThreads), 1, callStack);

	// Read the indices and head flags back to host memory.
	engine->indices.resize(4 * count);
	cuMemcpyDtoH(&engine->indices[0], sortData.values1[0], 4 * count);

	engine->flags.resize(count + 1);
	cuMemcpyDtoH(&engine->flags[0], pointers.headFlags, count);
	engine->flags[count] = 1;


	////////////////////////////////////////////////////////////////////////////
	// Iterate over the segment head flags. If there is an interval of two or
	// more elements in the same segment, then we have to sort them using 
	// memcmp. They aren't necessarily out of order, but the GPU radix sort 
	// didn't guarantee that they are in order.

	int numSegments = 0;
	int totalSegLength = 0;
	int prevSegStart = 0;
	QSortPred sortPred;
	sortPred.symbols = &engine->symbols[0];
	sortPred.count2 = count - gpuKeySize;
	sortPred.gpuKeySize = gpuKeySize;

	int* hostIndices = (int*)&engine->indices[0];

	for(int i(1); i <= count; ++i) {
		if(engine->flags[i]) {
			int len = i - prevSegStart;
			if(len >= 2) {
				++numSegments;
				totalSegLength += len;

		//		std::sort(hostIndices + prevSegStart, hostIndices + i, 
		//			sortPred);
			}
			prevSegStart = i;
		}
	}

	if(segCount) *segCount = numSegments;
	if(avSegSize) *avSegSize = numSegments ? 
		((float)totalSegLength / numSegments) : 0;

	if(indices) memcpy(indices, &engine->indices[0], 4 * count);
	
	if(transform) {
		byte* xform = (byte*)transform;
		for(int i(0); i < count; ++i) {
			int index = hostIndices[i] + count - 1;
			xform[i] = engine->symbols[index];
		}
	}

	return BWT_STATUS_SUCCESS;
}
