#include "../../../inc/mgpuscan.h"
#include "../../../util/cucpp.h"
#include "kernelparams.h"		// block size and density values.
#include <sstream>


struct KernelParams {
	int numThreads;
	int valuesPerThread;
	int blocksPerSM;
	const char* pass1;
	const char* pass2;
	const char* pass3;
};

const KernelParams Kernels[4] = {
	{ 
		SCAN_NUM_THREADS, 
		SCAN_VALUES_PER_THREAD,
		SCAN_BLOCKS_PER_SM,
		"GlobalScanUpsweep",
		"GlobalScanReduction",
		"GlobalScanDownsweep"
	}, {
		PACKED_NUM_THREADS,
		PACKED_VALUES_PER_THREAD,
		PACKED_BLOCKS_PER_SM,
		"SegScanUpsweepPacked",
		"SegScanReduction",
		"SegScanDownsweepPacked"
	}, {
		FLAGS_NUM_THREADS,
		FLAGS_VALUES_PER_THREAD,
		FLAGS_BLOCKS_PER_SM,
		"SegScanUpsweepFlags",
		"SegScanReduction",
		"SegScanDownsweepFlags"
	}, {
		KEYS_NUM_THREADS,
		KEYS_VALUES_PER_THREAD,
		KEYS_BLOCKS_PER_SM,
		"SegScanUpsweepKeys",
		"SegScanReduction",
		"SegScanDownsweepKeys"
	}
};



////////////////////////////////////////////////////////////////////////////////

const char* ScanStatusStrings[] = {
	"SCAN_STATUS_SUCCESS",
	"SCAN_STATUS_NOT_INITIALIZED",
	"SCAN_STATUS_DEVICE_ALLOC_FAILED",
	"SCAN_STATUS_INVALID_CONTEXT",
	"SCAN_STATUS_KERNEL_NOT_FOUND",
	"SCAN_STATUS_KERNEL_ERROR",
	"SCAN_STATUS_LAUNCH_ERROR",
	"SCAN_STATUS_INVALID_VALUE",
	"SCAN_STATUS_DEVICE_ERROR",
	"SCAN_STATUS_UNSUPPORTED_DEVICE"
};

const char* SCANAPI scanStatusString(scanStatus_t status) {
	int code = (int)status;
	if(code >= (int)(sizeof(ScanStatusStrings) / sizeof(char*))) return 0;
	return ScanStatusStrings[code];
}



struct scanEngine_d {
	ContextPtr context;

	ModulePtr module;

	FunctionPtr funcs[4][3];
	int numBlocks[4];
	int blockSize[4];

	DeviceMemPtr blockScanMem;
	DeviceMemPtr headFlagsMem;
	DeviceMemPtr rangeMem;

	std::vector<int2> ranges;
};


////////////////////////////////////////////////////////////////////////////////
// Initialize the scan engine.

scanStatus_t SCANAPI scanCreateEngine(const char* cubin, scanEngine_t* engine) {
	std::auto_ptr<scanEngine_d> e(new scanEngine_d);
	
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_INVALID_CONTEXT;

	if(2 != e->context->Device()->ComputeCapability().first)
		return SCAN_STATUS_UNSUPPORTED_DEVICE;

	result = e->context->LoadModuleFilename(cubin, &e->module);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_NOT_FOUND;

	int maxBlocks = 0;
	for(int i = 0; i < 4; ++i) {
		CUresult result = e->module->GetFunction(Kernels[i].pass1,
			make_int3(Kernels[i].numThreads, 1, 1), &e->funcs[i][0]);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

		result = e->module->GetFunction(Kernels[i].pass2,
			make_int3(REDUCTION_NUM_THREADS, 1, 1), &e->funcs[i][1]);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

		result = e->module->GetFunction(Kernels[i].pass3,
			make_int3(Kernels[i].numThreads, 1, 1), &e->funcs[i][2]);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

		int numSMs = e->context->Device()->Attributes().multiprocessorCount;

		e->numBlocks[i] = numSMs * Kernels[i].blocksPerSM;
		e->blockSize[i] = Kernels[i].numThreads * Kernels[i].valuesPerThread;

		maxBlocks = std::max(maxBlocks, e->numBlocks[i]);
	}

	// Allocate a uint per thread block plus a uint for the scan total.
	result = e->context->MemAlloc<uint>(1025 /*maxBlocks + 1*/, &e->blockScanMem);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_DEVICE_ALLOC_FAILED;

	result = e->context->MemAlloc<uint2>(maxBlocks, &e->rangeMem);
	result = e->context->MemAlloc<uint>(1025, &e->headFlagsMem);
	
	// Poke a zero to the start of the blockScanMem array. If we scan only a 
	// single block, this lets us skip the reduction step.
	int zero = 0;
	e->blockScanMem->FromHost(&zero, 4);

	*engine = e.release();
	return SCAN_STATUS_SUCCESS;
}

scanStatus_t SCANAPI scanDestroyEngine(scanEngine_t engine) {
	if(engine) delete engine;
	return SCAN_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Finds the ranges for each block to process. Note that each range must begin
// a multiple of the block size.

int SetBlockRanges(scanEngine_t engine, int count, int kind) {

	int numBlocks = engine->numBlocks[kind];
	int blockSize = engine->blockSize[kind];
	int numBricks = DivUp(count, blockSize);
	if(numBlocks > numBricks) numBlocks = numBricks;
	engine->ranges.resize(numBlocks);

	// Distribute the work along complete bricks.
	div_t brickDiv = div(numBricks, numBlocks);

	// Distribute the work along complete bricks.
	for(int i(0); i < numBlocks; ++i) {
		int2 range;
		range.x = i ? engine->ranges[i - 1].y : 0;
		int bricks = (i < brickDiv.rem) ? (brickDiv.quot + 1) : brickDiv.quot;
		range.y = std::min(range.x + bricks * blockSize, count);
		engine->ranges[i] = range;
	}
	engine->rangeMem->FromHost(engine->ranges);
	
	return (int)engine->ranges.size();
}


////////////////////////////////////////////////////////////////////////////////
// Generic scan - single implementation for all four scan types. Switches 
// over scan types when building the kernel's call stack.

scanStatus_t scanGeneric(scanEngine_t e, int kind, CUdeviceptr values,
	CUdeviceptr flags, CUdeviceptr scan, int count, uint* scanTotal,
	bool inclusive) {

	if(!e) return SCAN_STATUS_INVALID_VALUE;

	int numBlocks = SetBlockRanges(e, count, kind);

	CuCallStack callStack;
	CUresult result;

	if(numBlocks > 1) {

		////////////////////////////////////////////////////////////////////////
		// UPSWEEP
		// Don't run the upsweep on the last block - it contributes nothing to
		// reduction.

		switch(kind) {
			case 0:
				callStack.Push(values, e->blockScanMem, e->rangeMem);
				break;
			case 1:
				callStack.Push(values, e->blockScanMem, e->headFlagsMem,
					e->rangeMem);
				break;
			case 2:
			case 3:
				callStack.Push(values, flags, e->blockScanMem, e->headFlagsMem,
					e->rangeMem);
				break;
		}
		result = e->funcs[kind][0]->Launch(numBlocks - 1, 1, callStack);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

		
		////////////////////////////////////////////////////////////////////////
		// REDUCTION
		// Run a reduction for the block offsets if numBlocks > 1. We've already 
		// poked a 0 to the start of blockScanMem, so we can pass this step in
		// the case of a single block scan.

		callStack.Reset();
		if(0 == kind)
			callStack.Push(e->blockScanMem, numBlocks);
		else
			callStack.Push(e->headFlagsMem, e->blockScanMem, numBlocks);

		result = e->funcs[kind][1]->Launch(1, 1, callStack);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;
	}

	// Retrieve the scan total from the end of blockScanMem.
	if(scanTotal)
		e->blockScanMem->ToHostByte(scanTotal, sizeof(uint) * numBlocks,
			sizeof(uint));


	////////////////////////////////////////////////////////////////////////////
	// DOWNSWEEP
	
	callStack.Reset();
	switch(kind) {
		case 0:
		case 1:
			callStack.Push(values, scan, e->blockScanMem, e->rangeMem, count,
				(int)inclusive);
			break;
		case 2:
		case 3:
			callStack.Push(values, flags, scan, e->blockScanMem, e->rangeMem,
				count, (int)inclusive);
			break;
	}
	result = e->funcs[kind][2]->Launch(numBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

	return SCAN_STATUS_SUCCESS;
}


scanStatus_t SCANAPI scanArray(scanEngine_t engine, CUdeviceptr values,
	CUdeviceptr scan, int count, uint* scanTotal, bool inclusive) {

	return scanGeneric(engine, 0, values, 0, scan, count, scanTotal, inclusive);
}

scanStatus_t SCANAPI scanSegmentedPacked(scanEngine_t engine,
	CUdeviceptr packed, CUdeviceptr scan, int count, bool inclusive) {

	return scanGeneric(engine, 1, packed, 0, scan, count, 0, inclusive);
}

scanStatus_t SCANAPI scanSegmentedFlags(scanEngine_t engine, CUdeviceptr values,
	CUdeviceptr flags, CUdeviceptr scan, int count, bool inclusive) {

	return scanGeneric(engine, 2, values, flags, scan, count, 0, inclusive);
}

scanStatus_t SCANAPI scanSegmentedKeys(scanEngine_t engine, CUdeviceptr values,
	CUdeviceptr keys, CUdeviceptr scan, int count, bool inclusive) {
	
	return scanGeneric(engine, 3, values, keys, scan, count, 0, inclusive);
}
