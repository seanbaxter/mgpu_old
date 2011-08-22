#include "sortdll.h"
#include "../../inc/mgpusort.hpp"
#include "../kernels/params.cu"
#include <sstream>

const char* SortStatusStrings[] = {
	"SORT_STATUS_SUCCESS",
	"SORT_STATUS_NOT_INITIALIZED",
	"SORT_STATUS_DEVICE_ALLOC_FAILED",
	"SORT_STATUS_HOST_ALLOC_FAILED",
	"SORT_STATUS_CONFIG_NOT_SUPPORTED",
	"SORT_STATUS_CONTEXT_MISMATCH",
	"SORT_STATUS_INVALID_CONTEXT",
	"SORT_STATUS_KERNEL_NOT_FOUND",
	"SORT_STATUS_KERNEL_ERROR",
	"SORT_STATUS_LAUNCH_ERROR",
	"SORT_STATUS_INVALID_VALUE",
	"SORT_STATUS_DEVICE_ERROR",
	"SORT_STATUS_INTERNAL_ERROR",
	"SORT_STATUS_UNSUPPORTED_DEVICE"
};

const char* SORTAPI sortStatusString(sortStatus_t status) {
	int code = (int)status;
	if(code >= sizeof(SortStatusStrings) / sizeof(char*)) return 0;
	return SortStatusStrings[code];
}

const int NumCountWarps = NUM_COUNT_WARPS;
const int NumCountThreads = WarpSize * NUM_COUNT_WARPS;

const int NumHistWarps = NUM_HIST_WARPS;
const int NumHistThreads = WarpSize * NUM_HIST_WARPS;

const int SORT_END_KEY_SAVE = 1;
const int SORT_END_KEY_SET = 2;
const int SORT_END_KEY_RESTORE = 4;



////////////////////////////////////////////////////////////////////////////////
// LoadCountModule, LoadHistModule, LoadSortModule

bool LoadFunctions(const char* name, int numThreads, CuModule* module,
	FunctionPtr functions[6]) {

		// TODO: change to bits = 1
	for(int bits(1); bits <= 6; ++bits) {
		std::ostringstream oss;
		oss<< name<< "_"<< bits;
		CUresult result = module->GetFunction(oss.str(), 
			make_int3(numThreads, 1, 1), &functions[bits - 1]);
		if(CUDA_SUCCESS != result) return false;
	}
	return true;
}

sortStatus_t LoadCountModule(sortEngine_d* engine, const char* path, 
	std::auto_ptr<sortEngine_d::CountKernel>* count) {

	std::auto_ptr<sortEngine_d::CountKernel> c(new sortEngine_d::CountKernel);

	std::ostringstream oss;
	oss<< path<< "count.cubin";
	
	CUresult result = engine->context->LoadModuleFilename(oss.str(), 
		&c->module);
	if(CUDA_SUCCESS != result) return SORT_STATUS_KERNEL_NOT_FOUND;
	
	bool success = LoadFunctions("CountBuckets", NumCountThreads, c->module,
		c->functions);
	if(!success) return SORT_STATUS_KERNEL_ERROR;

	success = LoadFunctions("CountBuckets_ee", NumCountThreads, c->module, 
		c->eeFunctions);
	if(!success) return SORT_STATUS_KERNEL_ERROR;

	*count = c;
	return SORT_STATUS_SUCCESS;
}

sortStatus_t LoadHistModule(sortEngine_d* engine, const char* path, 
	bool transList, std::auto_ptr<sortEngine_d::HistKernel>* hist) {

	std::auto_ptr<sortEngine_d::HistKernel> h(new sortEngine_d::HistKernel);

	std::ostringstream oss;
	oss<< path<< "hist_"<< (transList ? "list" : "simple") << ".cubin";

	CUresult result = engine->context->LoadModuleFilename(oss.str(),
		&h->module);
	if(CUDA_SUCCESS != result) return SORT_STATUS_KERNEL_NOT_FOUND;

	bool success = LoadFunctions("HistogramReduce_1", NumHistThreads, 
		h->module, h->pass1);
	if(success) 
		success = LoadFunctions("HistogramReduce_2", NumHistThreads, h->module,
			h->pass2);
	if(success) 
		success = LoadFunctions("HistogramReduce_3", NumHistThreads, h->module, 
			h->pass3);
	if(!success) return SORT_STATUS_KERNEL_ERROR;

	*hist = h;
	return SORT_STATUS_SUCCESS;
}

sortStatus_t LoadSortModule(sortEngine_d* engine, const char* path, 
	int numThreads, int valuesPerThread, int valueCode, bool transList, 
	std::auto_ptr<sortEngine_d::SortKernel>* sort) {

	std::auto_ptr<sortEngine_d::SortKernel> s(new sortEngine_d::SortKernel);

	std::ostringstream oss;
	oss<< path<< "sort_"<< numThreads<< "_"<< valuesPerThread<< "_";
	switch(valueCode) {
		case 0: oss<< "key"; break;
		case 1: oss<< "index"; break;
		case 2: oss<< "single"; break;
		case 3: oss<< "multi"; break;
	}
	oss<< "_"<< (transList ? "list" : "simple")<< ".cubin";
	
	CUresult result = engine->context->LoadModuleFilename(oss.str(), 
		&s->module);
	if(CUDA_SUCCESS != result) return SORT_STATUS_KERNEL_NOT_FOUND;
	
	bool success = LoadFunctions("RadixSort", numThreads, s->module, 
		s->functions);
	if(!success) return SORT_STATUS_KERNEL_ERROR;

	success = LoadFunctions("RadixSort_ee", numThreads, s->module, 
		s->eeFunctions);
	if(!success) return SORT_STATUS_KERNEL_ERROR;

	*sort = s;
	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// LoadKernels, PreloadKernels (export)

sortStatus_t LoadKernels(sortEngine_t engine, int numSortThreads, 
	int valuesPerThread, bool useTransList, int valueCode, 
	sortEngine_d::HistKernel** pHist, sortEngine_d::SortKernel** pSort) {

	if(128 != numSortThreads && 256 != numSortThreads)
		return SORT_STATUS_INVALID_VALUE;
	if(4 != valuesPerThread && 8 != valuesPerThread)
		return SORT_STATUS_INVALID_VALUE;
	if(valueCode < 0 || valueCode > 4)
		return SORT_STATUS_INVALID_VALUE;

	// Load the hist kernel.
	std::auto_ptr<sortEngine_d::HistKernel>& hist = engine->hist[useTransList];
	if(!hist.get()) {
		sortStatus_t status = LoadHistModule(engine, engine->kernelPath.c_str(), 
			useTransList, &hist);
		if(SORT_STATUS_SUCCESS != status) return status;
	}
	if(pHist) *pHist = hist.get();

	// Load the sort kernel.
	int numThreadsCode = 256 == numSortThreads;
	int valCode = 8 == valuesPerThread;
	std::auto_ptr<sortEngine_d::SortKernel>& sort = 
		engine->sort[useTransList][numThreadsCode][valCode][valueCode];
	if(!sort.get()) {
		sortStatus_t status = LoadSortModule(engine, engine->kernelPath.c_str(),
			numSortThreads, valuesPerThread, valueCode, useTransList, &sort);
		if(SORT_STATUS_SUCCESS != status) return status;
	}
	if(pSort) *pSort = sort.get();

	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortLoadKernel(sortEngine_t engine, int numSortThreads,
	int valuesPerThread, bool useTransList, int valueCount) {

	int valueCode = 0;
	if(valueCount >= 2) valueCode = 3;
	else if(0 != valueCount) valueCode = 2;

	sortStatus_t status = LoadKernels(engine, numSortThreads, valuesPerThread,
		useTransList, valueCode, 0, 0);

	if(-1 == valueCount && SORT_STATUS_SUCCESS == status)
		status = LoadKernels(engine, numSortThreads, valuesPerThread, 
			useTransList, 1, 0, 0);

	return status;
}


////////////////////////////////////////////////////////////////////////////////
// sortCreateEngine, sortDestroyEngine

sortStatus_t SORTAPI sortCreateEngine(const char* kernelPath,
	sortEngine_t* engine) {
	
	EnginePtr e(new sortEngine_d);
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return SORT_STATUS_INVALID_CONTEXT;
	if(e->context->Device()->ComputeCapability().first < 2)
		return SORT_STATUS_UNSUPPORTED_DEVICE;;

	e->kernelPath = kernelPath;
	e->numSMs = e->context->Device()->NumSMs();

	// Load the count module - there is just one for all permutations of the
	// sort, so load it here.
	sortStatus_t status = LoadCountModule(e, kernelPath, &e->count);
	if(SORT_STATUS_SUCCESS != status) return status;

	// If we need to set the end keys to -1 to eliminate range comparisons, it
	// may be necessary to first save those values here.
	result = e->context->MemAlloc<uint>(2048, &e->keyRestoreBuffer);
	if(CUDA_SUCCESS != result) return SORT_STATUS_HOST_ALLOC_FAILED;

	// Get a pointer to the global memory backing the sortDetectCounters.
	result = e->count->module->GetGlobal("sortDetectCounters_global", 
		&e->sortDetectCounters);
	if(CUDA_SUCCESS != result) return SORT_STATUS_KERNEL_ERROR;

	// Allocate a range pair for each hist warp launched.
	int numRangePairs = e->numSMs * NumHistWarps;
	e->rangePairsHost.resize(numRangePairs);

	result = e->context->ByteAlloc(sizeof(int2) * numRangePairs,
		&e->rangePairs);
	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;

	// Allocate a full set of buckets for each warp of the histogram kernels
	// launched.
	result = e->context->MemAlloc<uint>(e->numSMs * NumHistWarps * 64, 
		&e->columnScan);
	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;
	
	// Allocate a full set of buckets (64) for each hist kernel that may be
	// launched.
	result = e->context->MemAlloc<uint>(e->numSMs * 64, &e->countScan);
	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;

	e->lastNumHistRowsProcessed = 0;

	*engine = e.release();
	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortIncEngine(sortEngine_t engine) {
	if(!engine) return SORT_STATUS_INVALID_VALUE;
	static_cast<sortEngine_d*>(engine)->AddRef();
	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortReleaseEngine(sortEngine_t engine) {
	if(engine) static_cast<sortEngine_d*>(engine)->Release();
	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// SortTerms used both when sorting and when allocating engine resources.

struct SortTerms {
	int numSortBlocks;
	int numCountBlocks;
	int countValuesPerThread;
	int countSize;
	int numHistRows;
	int numHistBlocks;
	int scatterStructSize;
	int bucketCodesSize;
	int numEndKeys;
};

SortTerms ComputeSortTerms(int numSortThreads, int valuesPerThread, 
	bool useTransList, int numBits, int numElements, int numSMs) {

	SortTerms terms;
	
	int numValues = numSortThreads * valuesPerThread;
	terms.numSortBlocks = DivUp(numElements, numValues);
	terms.numCountBlocks = DivUp(terms.numSortBlocks, NumCountWarps);
	terms.countValuesPerThread = numValues / WarpSize;

	int numBuckets = 1<< numBits;
	terms.countSize = 4 * std::max(WarpSize, numBuckets * NumCountWarps) * 
		terms.numCountBlocks;

	int numChannels = numBuckets / 2;
	int numSortBlocksPerCountWarp = 
		std::min(NumCountWarps, WarpSize / numChannels);
	terms.numHistRows = DivUp(terms.numSortBlocks, numSortBlocksPerCountWarp);
	terms.numHistBlocks = std::min(numSMs, 
		DivUp(terms.numHistRows, NumHistWarps));

	int bucketCodeBlockSize = numBuckets;
	if(useTransList) bucketCodeBlockSize += numBuckets + WarpSize;
	terms.scatterStructSize = RoundUp(bucketCodeBlockSize, WarpSize);
	terms.bucketCodesSize = 4 * terms.numSortBlocks * terms.scatterStructSize;

	terms.numEndKeys = RoundUp(numElements, numValues) - numElements;

	return terms;
}


////////////////////////////////////////////////////////////////////////////////
// AllocSortResources

sortStatus_t AllocSortResources(const SortTerms& terms, sortEngine_t engine) {

	// The count kernel will pack counters together to optimize space. However
	// it can't pack more than numCountWarps blocks into a single segment
	// (32 values).
	if(!engine->countBuffer.get() || 
		(terms.countSize > (int)engine->countBuffer->Size())) {
		DeviceMemPtr mem;
		CUresult result = engine->context->ByteAlloc(terms.countSize, &mem);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;
		engine->countBuffer = mem;
	}

	// Allocate space in the engine object for the bucket codes.
	if(!engine->bucketCodes.get() || 
		terms.bucketCodesSize > (int)engine->bucketCodes->Size()) {
		DeviceMemPtr mem;
		CUresult result = engine->context->ByteAlloc(terms.bucketCodesSize,
			&mem);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;
		engine->bucketCodes = mem;
	}

	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Define sortArrays in terms of sortPass

// Sort a single pass. The arguments override the members in sortData_d.
// earlyExitCode is:
// 0 - no early exit
// 1 - some blocks are radix sorted. use sort kernel with early exit.
// 2 - array is globally sorted over radix digit. proceed to next radix digit.
// 3 - array is globally sorted over full keys. the sort is done.

sortStatus_t sortPass(sortEngine_t engine, sortData_t data, int numSortThreads, 
	int valuesPerThread, bool useTransList, int firstBit, int endBit, 
	int endKeyFlags, int valueCode, int* earlyExitCode, int& parity) {

	if(data->numElements > data->maxElements) return SORT_STATUS_INVALID_VALUE;

	if((firstBit < 0) || (endBit > 32) || (endBit <= firstBit) || 
		((endBit - firstBit) > 6))
		return SORT_STATUS_INVALID_VALUE;
	
	int numBits = endBit - firstBit;

	SortTerms terms = ComputeSortTerms(numSortThreads, valuesPerThread,
		useTransList, numBits, data->numElements, engine->numSMs);

	sortEngine_d::HistKernel* hist;
	sortEngine_d::SortKernel* sort;
	CUresult result;
	sortStatus_t status = LoadKernels(engine, numSortThreads, valuesPerThread,
		useTransList, valueCode, &hist, &sort);
	if(SORT_STATUS_SUCCESS != status) return status;

	status = AllocSortResources(terms, engine);
	if(SORT_STATUS_SUCCESS != status) return status;
	
	// Set numHistRows into rangePairs if it hasn't already been set to this 
	// size.
	if(terms.numHistRows != engine->lastNumHistRowsProcessed) {
		int2* pairs = &engine->rangePairsHost[0];
		int numPairs = terms.numHistBlocks * NumHistWarps;
		int pairCount = terms.numHistRows / numPairs;
		int pairSplit = terms.numHistRows % numPairs;
		pairs[0].x = 0;
		for(int i = 0; i < numPairs; ++i) {
			if(i) pairs[i].x = pairs[i - 1].y;
			pairs[i].y = pairs[i].x + pairCount + (i < pairSplit);
		}

		// Copy rangePairsHost to device memory.
		CUresult result = engine->rangePairs->FromHost(
			&engine->rangePairsHost[0], numPairs);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;

		engine->lastNumHistRowsProcessed = terms.numHistRows;
	}

	// Save the trailing keys
	if((SORT_END_KEY_SAVE & endKeyFlags) && terms.numEndKeys) {
		engine->restoreSourceSize = terms.numEndKeys;
		CUdeviceptr source = AdjustPointer<uint>(data->keys[0],
			data->numElements);
		CUresult result = cuMemcpy(engine->keyRestoreBuffer->Handle(), source, 
			4 * engine->restoreSourceSize);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
	}

	// Set the trailing keys to all set bits here.
	if((SORT_END_KEY_SET & endKeyFlags) && terms.numEndKeys) {
		// Back up the overwritten keys in the engine
		CUdeviceptr target = AdjustPointer<uint>(data->keys[0],
			data->numElements);
		CUresult result = cuMemsetD32(target, 0xffffffff, terms.numEndKeys);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
	}

	// Run the count kernel
	if(data->earlyExit) engine->sortDetectCounters->Fill(0);

	CuCallStack callStack;
	callStack.Push(data->keys[0], firstBit, data->numElements, 
		terms.countValuesPerThread, engine->countBuffer);
	CuFunction* count = data->earlyExit ? 
		engine->count->eeFunctions[numBits - 1].get() :
		engine->count->functions[numBits - 1].get();
	result = count->Launch(terms.numCountBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SORT_STATUS_LAUNCH_ERROR;

	*earlyExitCode = 0;
	if(data->earlyExit) {
		uint4 detect;
		result = engine->sortDetectCounters->ToHost(&detect, 16);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;

		uint radixSort = detect.x;
		uint fullCount = detect.y;
		uint radixCount = detect.z;

		if(terms.numCountBlocks == fullCount) *earlyExitCode = 3;
		else if(terms.numCountBlocks == radixCount) *earlyExitCode = 2;

		// If 5% of the sort blocks are sorted, use the slightly slower early 
		// exit sort kernel.
		else if((double)radixSort / terms.numSortBlocks > 0.05)
			*earlyExitCode = 1;
		else *earlyExitCode = 0;
	}

	if(*earlyExitCode <= 1) {

		// Run the three histogram kernels
		callStack.Reset();
		callStack.Push(engine->countBuffer, engine->rangePairs, 
			engine->countScan, engine->columnScan);
		result = hist->pass1[numBits - 1]->Launch(terms.numHistBlocks, 1,
			callStack);
		if(CUDA_SUCCESS != result) return SORT_STATUS_LAUNCH_ERROR;

		callStack.Reset();
		callStack.Push(terms.numHistBlocks, engine->countScan);
		result = hist->pass2[numBits - 1]->Launch(1, 1, callStack);
		if(CUDA_SUCCESS != result) return SORT_STATUS_LAUNCH_ERROR;

		callStack.Reset();
		callStack.Push(engine->countBuffer, engine->rangePairs, 
			engine->countScan, engine->columnScan, engine->bucketCodes,
			*earlyExitCode);
		result = hist->pass3[numBits - 1]->Launch(terms.numHistBlocks, 1,
			callStack);
		if(CUDA_SUCCESS != result) return SORT_STATUS_LAUNCH_ERROR;

		// Run the sort kernel
		callStack.Reset();
		callStack.Push(data->keys[0], engine->bucketCodes, firstBit, 
			data->keys[1]);
		
		switch(valueCode) {
			case 1:		// VALUE_TYPE_INDEX
				callStack.Push(data->values1[1]); 
				break;
			case 2:		// VALUE_TYPE_SINGLE
				callStack.Push(data->values1[0], data->values1[1]);
				break;
			case 3:		// VALUE_TYPE_MULTI
				callStack.Push(data->valueCount,
					// Six values_global_in
					data->values1[0], data->values2[0], data->values3[0],
					data->values4[0], data->values5[0], data->values6[0],
					
					// Six values_global_out
					data->values1[1], data->values2[1], data->values3[1],
					data->values4[1], data->values5[1], data->values6[1]);
				break;
		}

		CuFunction* sortFunc = *earlyExitCode ? 
			sort->eeFunctions[numBits - 1].get() :
			sort->functions[numBits - 1].get();

		result = sortFunc->Launch(terms.numSortBlocks, 1, callStack);
		if(CUDA_SUCCESS != result) return SORT_STATUS_LAUNCH_ERROR;

		// Swap the source and target buffers in the data structure.
		std::swap(data->keys[0], data->keys[1]);
		std::swap(data->values1[0], data->values1[1]);
		std::swap(data->values2[0], data->values2[1]);
		std::swap(data->values3[0], data->values3[1]);
		std::swap(data->values4[0], data->values4[1]);
		std::swap(data->values5[0], data->values5[1]);
		std::swap(data->values6[0], data->values6[1]);
		parity ^= 1;
	}

	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// GetSortCode, sortArrayFromList

// sortArrayFromList calls sortPass2 for each pass in a sort list. This list can
// be pulled from a pre-computed sort table (for sortArray) or generated to 
// favor a particular kind of pass (sortArrayEx).

// Returns endKeyFlags in .first and valueCode in .second.
IntPair GetSortCode(int bit, int endBit, sortData_t data) {

	int endKeyFlags = 0;
	if(bit == data->firstBit) {
		endKeyFlags |= SORT_END_KEY_SET;
		if(data->preserveEndKeys) endKeyFlags |= SORT_END_KEY_SAVE;
	}

	int valueCode;
	switch(data->valueCount) {
		case 0: valueCode = 0; break;
		case 1: valueCode = 2; break;
		case -1: valueCode = (bit == data->firstBit) ? 1 : 2; break;
		default: valueCode = 3; break;
	}
	
	return IntPair(endKeyFlags, valueCode);
}

sortStatus_t sortArrayFromList(sortEngine_t engine, sortData_t data, 
	int numSortThreads, int valuesPerThread, bool useTransList, 
	const char* table) {

	int bit = data->firstBit;

	// Clear the restore size counter at the start of the sort. If this is 
	// non-zero at the end of the sort (even if the sort failed), copy from the
	// restore buffer to the buffer that was at keys[0] when the sorting began.
	engine->restoreSourceSize = 0;
	CUdeviceptr firstKeysBuffer = data->keys[0];
	sortStatus_t status;

	// Loop through each element of the pass table.
	for(int i(0); i < 6; ++i)
		for(int j(0); j < table[i]; ++j) {
			int endBit = bit + i + 1;

			IntPair sortCode = GetSortCode(bit, endBit, data);
			int earlyExit;
			status = sortPass(engine, data, numSortThreads, valuesPerThread,
				useTransList, bit, endBit, sortCode.first, sortCode.second,
				&earlyExit, data->parity);
			if(SORT_STATUS_SUCCESS != status) break;
			
			bit = endBit;
			if(2 == earlyExit) { i = 6; break; }
		}

	// Restore the trailing keys.
	if(engine->restoreSourceSize) {
		CUdeviceptr target = AdjustPointer<uint>(firstKeysBuffer,
			data->numElements);
		CUresult result = cuMemcpy(target, engine->keyRestoreBuffer->Handle(), 
			4 * engine->restoreSourceSize);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
	}
	
	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sortArray, sortArrayEx - mgpusort export functions

sortStatus_t SORTAPI sortArray(sortEngine_t engine, sortData_t data) {
	// Find the optimal sort sequence from a table.
	int numBits = data->endBit - data->firstBit;
	if(!numBits) return SORT_STATUS_SUCCESS;
	if(numBits > 32) return SORT_STATUS_INVALID_VALUE;

	const char* table;
	int numThreads;
	if(0 == data->valueCount) {
		// Key-only, 128 threads.
		table = sort_128_8_key_simple_table[numBits - 1];
		numThreads = 128;
	} else if(1 == abs(data->valueCount)) {
		// Single value, 256 threads.
		table = sort_256_8_single_simple_table[numBits - 1];
		numThreads = 256;
	} else if(2 == data->valueCount) {
		// TODO: generate remaining tables.
	} 

	return sortArrayFromList(engine, data, numThreads, 8, false, table);
}

sortStatus_t SORTAPI sortArrayEx(sortEngine_t engine, sortData_t data, 
	int numSortThreads, int valuesPerThread, int bitPass, bool useTransList) {

	if((128 != numSortThreads && 256 != numSortThreads) || 
		(8 != valuesPerThread))
		return SORT_STATUS_INVALID_VALUE;

	if(data->numElements > data->maxElements) return SORT_STATUS_INVALID_VALUE;
	if(bitPass <= 0 || bitPass > 6) bitPass = 6;

	int numBits = data->endBit - data->firstBit;
	if(bitPass > numBits) bitPass = numBits;

	int numPasses = DivUp(numBits, bitPass);
	int split = numPasses * bitPass - numBits;

	// Generate a pass list.
	char list[6] = { 0 };
	int bit = data->firstBit;

	for(int pass(0); pass < numPasses; ++pass) {
		numBits = bitPass - (pass < split);
		++list[numBits - 1];
		bit += numBits;
	}

	return sortArrayFromList(engine, data, numSortThreads, valuesPerThread, 
		useTransList, list); 
}

		

////////////////////////////////////////////////////////////////////////////////
// sortCreateData, sortDestroyData

sortStatus_t SORTAPI sortCreateData(sortEngine_t engine, int maxElements, 
	int valueCount, sortData_t* data) {

	if(valueCount > 6) return SORT_STATUS_INVALID_VALUE;

	std::auto_ptr<sortData_d> d(new sortData_d);

	// sortData_d
	d->maxElements = RoundUp(maxElements, 2048);
	d->numElements = maxElements;
	d->valueCount = valueCount;
	d->firstBit = 0;
	d->endBit = 32;
	d->preserveEndKeys = false;
	d->earlyExit = false;
	d->keys[0] = d->keys[1] = 0;
	d->values1[0] = d->values1[1] = 0;
	d->values2[0] = d->values2[1] = 0;
	d->values3[0] = d->values3[1] = 0;
	d->values4[0] = d->values4[1] = 0;
	d->values5[0] = d->values5[1] = 0;
	d->values6[0] = d->values6[1] = 0;
	
	sortAllocData(engine, d.get());

	*data = d.release();
	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortDestroyData(sortData_t data) {
	if(!data) return SORT_STATUS_SUCCESS;
	sortFreeData(data);
	delete static_cast<sortData_d*>(data);
	
	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortAllocData(sortEngine_t engine, sortData_t data) {
	if(data->valueCount > 6) return SORT_STATUS_INVALID_VALUE;

	DeviceMemPtr mem[7 * 2];
	uint maxElements = RoundUp(data->maxElements, 2048);
	int count = (-1 == data->valueCount) ? 1 : data->valueCount;

	CUresult result = CUDA_SUCCESS;

	CUdeviceptr* memPtr = data->keys;
	for(int i = 0; (i < 2 + 2 * count) && (CUDA_SUCCESS == result); ++i) {
		if(!memPtr[i]) {
			result = engine->context->MemAlloc<uint>(maxElements, &mem[i]);
			if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;
		}
	}

	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ALLOC_FAILED;

	for(int i(0); i < 7 * 2; ++i)
		if(mem[i]) {
			memPtr[i] = mem[i]->Handle();
			mem[i].release();
		}

	data->parity = 0;
	return SORT_STATUS_SUCCESS;
}

sortStatus_t SORTAPI sortFreeData(sortData_d* data) {
	if(!data) return SORT_STATUS_INVALID_VALUE;
	for(int i(0); i < 2; ++i) {
		cuFreeZero(data->keys[i]);
		cuFreeZero(data->values1[i]);
		cuFreeZero(data->values2[i]);
		cuFreeZero(data->values3[i]);
		cuFreeZero(data->values4[i]);
		cuFreeZero(data->values5[i]);
		cuFreeZero(data->values6[i]);
	}
	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sortHost

sortStatus_t SORTAPI sortHost(sortEngine_t engine, uint* keys, uint* values,
	int numElements, int numBits) {

	MgpuSortData data;
	sortStatus_t status = data.Alloc(engine, numElements, values ? 1 : 0);
	if(SORT_STATUS_SUCCESS != status) return status;

	data.endBit = numBits;

	// Copy keys and values into device memory.
	CUresult result = cuMemcpyHtoD(data.keys[0], keys, 
		sizeof(double) * numElements);
	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;

	if(values) {
		result = cuMemcpyHtoD(data.values1[0], values, 
			sizeof(double) * numElements);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
	}

	// Sort
	status = sortArray(engine, &data);
	if(SORT_STATUS_SUCCESS != status) return status;

	// Copy sorted keys and values into host memory.
	result = cuMemcpyDtoH(keys, data.keys[0], sizeof(double) * numElements);
	if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;

	if(values) {
		result = cuMemcpyDtoH(values, data.values1[0], 
			sizeof(double) * numElements);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
	}

	// MgpuSortData wrapper will automatically clean up the device memory.
	return SORT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sortDevice

sortStatus_t SORTAPI sortDevice(sortEngine_t engine, CUdeviceptr keys, 
	CUdeviceptr values, int numElements, int numBits) {

	MgpuSortData data;
	data.AttachKey(keys);
	if(values) data.AttachVal(0, values);
	sortStatus_t status = data.Alloc(engine, numElements, values ? 1 : 0);
	if(SORT_STATUS_SUCCESS != status) return status;

	data.endBit = numBits;

	status = sortArray(engine, &data);
	if(SORT_STATUS_SUCCESS != status) return status;

	if(data.parity) {
		CUresult result = cuMemcpyDtoD(keys, data.keys[1],
			sizeof(uint) * numElements);
		if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
		if(values) {
			cuMemcpyDtoD(values, data.values1[1], sizeof(uint) * numElements);
			if(CUDA_SUCCESS != result) return SORT_STATUS_DEVICE_ERROR;
		}
	}
	return SORT_STATUS_SUCCESS;
}

