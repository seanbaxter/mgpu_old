#include "../../../inc/mgpuselect.h"
#include "../../../util/cucpp.h"
#include <sstream>

const char* CountKernels[3] = { 
	"KSmallestCountUint",
	"KSmallestCountInt",
	"KSmallestCountFloat"
};
const char* HistKernels[2] = {
	"KSmallestHistValue",
	"KSmallestHistInterval"
};
const char* StreamKernels[3][3][3] = {
	{
		{
			"KSmallestStreamUintA",
			"KSmallestStreamIntA",
			"KSmallestStreamFloatA"
		}, {
			"KSmallestStreamUintAB",
			"KSmallestStreamIntAB",
			"KSmallestStreamFloatAB"
		}, {
			"KSmallestStreamUintInterval",
			"KSmallestStreamIntInterval",
			"KSmallestStreamFloatInterval"
		}
	}, {
		{
			"KSmallestStreamUintGenA",
			"KSmallestStreamUintGenA",
			"KSmallestStreamFloatGenA"
		}, {
			"KSmallestStreamUintGenAB",
			"KSmallestStreamIntGenB",
			"KSmallestStreamFloatGenAB"
		}, {
			"KSmallestStreamUintGenInterval",
			"KSmallestStreamIntGenInterval",
			"KSmallestStreamFloatGenInterval"
		}
	}, {
		{
			"KSmallestStreamUintPairA",
			"KSmallestStreamIntPairA",
			"KSmallestStreamFloatPairA"
		}, {
			"KSmallestStreamUintPairAB",
			"KSmallestStreamIntPairAB",
			"KSmallestStreamFloatPairAB"
		}, {
			"KSmallestStreamUintPairInterval",
			"KSmallestStreamIntPairInterval",
			"KSmallestStreamFloatPairInterval"
		}
	}
};


const char* SelectStatusStrings[] = {
	"SELECT_STATUS_SUCCESS",
	"SELECT_STATUS_NOT_INITIALIZED",
	"SELECT_STATUS_DEVICE_ALLOC_FAILED",
	"SELECT_STATUS_INVALID_CONTEXT",
	"SELECT_STATUS_KERNEL_NOT_FOUND",
	"SELECT_STATUS_KERNEL_ERROR",
	"SELECT_STATUS_LAUNCH_ERROR",
	"SELECT_STATUS_INVALID_VALUE",
	"SELECT_STATUS_DEVICE_ERROR",
	"sELECT_STATUS_UNSUPPORTED_DEVICE"
};

// Returns a textual representation of the enums.
const char* SELECTAPI selectStatusString(selectStatus_t status) {
	int code = (int)status;
	if(code >= (int)(sizeof(SelectStatusStrings) / sizeof(char*))) return 0;
	return SelectStatusStrings[code];
}


struct selectEngine_d {
	ContextPtr context;
	ModulePtr module;

	// Index is type (uint, int, float)
	FunctionPtr countFuncs[3];

	// Index is single/interval.
	FunctionPtr histFuncs[2];

	FunctionPtr streamFuncs[3][3][3];

	int blockSize;
	int warpsPerBlock;
	int blocksPerSM;
	int warpsPerSM;
	int numWarps;			// total number of warps.
	int warpValues;			// min values processed per warp.

	DeviceMemPtr countMem;
	DeviceMemPtr scanTotalMem;
	DeviceMemPtr scanWarpMem;
	DeviceMemPtr rangeMem;

	// One pair of ranges per warp.
	std::vector<int2> ranges;
};


////////////////////////////////////////////////////////////////////////////////
// Initialize the select engine.

selectStatus_t SELECTAPI selectCreateEngine(const char* kernelPath,
	selectEngine_t* engine) {

	std::auto_ptr<selectEngine_d> e(new selectEngine_d);

	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_INVALID_CONTEXT;

	if(2 != e->context->Device()->ComputeCapability().first)
		return SELECT_STATUS_UNSUPPORTED_DEVICE;

	result = e->context->LoadModuleFilename(kernelPath, &e->module);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_KERNEL_NOT_FOUND;

	// These kernels run 6 blocks per SM and 4 warps per block.
	e->blockSize = 128;
	e->warpsPerBlock = e->blockSize / WarpSize;
	e->blocksPerSM = 6;
	e->warpsPerSM = e->warpsPerBlock * e->blocksPerSM;
	e->numWarps = e->warpsPerSM * e->context->Device()->NumSMs();
	
	// Allow for unrolled loops of 8 values per thread.
	e->warpValues = 8 * 32;

	// Load the count kernels.
	for(int i(0); i < 3; ++i) {
		result = e->module->GetFunction(CountKernels[i], make_int3(128, 1, 1),
			&e->countFuncs[i]);
		if(CUDA_SUCCESS != result) return SELECT_STATUS_KERNEL_ERROR;
	}

	// Load the hist kernels.
	for(int i(0); i < 2; ++i) {
		result = e->module->GetFunction(HistKernels[i], make_int3(1024, 1, 1),
			&e->histFuncs[i]);
		if(CUDA_SUCCESS != result) return SELECT_STATUS_KERNEL_ERROR;
	}



	// Load the stream kernels.
	result = e->module->GetFunction("KSmallestStreamValue", 
		make_int3(128, 1, 1), &e->streamFuncs[0][0][0]);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_KERNEL_ERROR;



	// Allocate a range pair for each warp that can be launched.
	e->ranges.resize(e->numWarps);
	result = e->context->MemAlloc<int2>(e->numWarps, &e->rangeMem);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_DEVICE_ALLOC_FAILED;

	// Allocate 64 ints for counts for each warp.
	result = e->context->MemAlloc<uint>(64 * e->numWarps, &e->countMem);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_DEVICE_ALLOC_FAILED;
	
	// Allocate combined space for 
	// 1) Total counts for each bucket (64)
	// 2) Exc scan of bucket counts (64)
	// 3) k1 and k2 buckets (2)
	result = e->context->MemAlloc<uint>(130, &e->scanTotalMem);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_DEVICE_ALLOC_FAILED;
	
	// 3) Exc scan of k1 buckets over warp (numWarps)
	// 4) Exc scan of k2 buckets over warp (numWarps)
	// 5) Exc scan of middle buckets over warp (numWarps)
	result = e->context->MemAlloc<uint>(3 * RoundUp(e->numWarps, WarpSize),
		&e->scanWarpMem);
	if(CUDA_SUCCESS != result) return SELECT_STATUS_DEVICE_ALLOC_FAILED;

	*engine = e.release();
	return SELECT_STATUS_SUCCESS;
}


selectStatus_t SELECTAPI selectDestroyEngine(selectEngine_t engine) {
	if(engine) delete engine;
	return SELECT_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Finds the ranges for each block to process. Note that each range must begin
// a multiple of the block size.

int SetBlockRanges(selectEngine_t engine, int count) {

	int numWarps = engine->numWarps;
	int numBricks = DivUp(count, engine->warpValues);
	if(numBricks < numWarps) numWarps = numBricks;

	std::fill(engine->ranges.begin(), engine->ranges.end(), make_int2(0, 0));

	// Distribute the work evenly over all warps.
	div_t brickDiv = div(numBricks, numWarps);

	// Distribute the work along complete bricks.
	for(int i(0); i < numWarps; ++i) {
		int2 range;
		range.x = i ? engine->ranges[i - 1].y : 0;
		int bricks = (i < brickDiv.rem) ? (brickDiv.quot + 1) : brickDiv.quot;
		range.y = std::min(range.x + bricks * engine->warpValues, count);
		engine->ranges[i] = range;
	}
	CUresult result = engine->rangeMem->FromHost(engine->ranges);
	
	return numWarps;
}


////////////////////////////////////////////////////////////////////////////////
// selectValue

// Returns the k'th-smallest value. If index is not null, it also returns the
// position within the original array that the value appears at.
selectStatus_t SELECTAPI selectValue(selectEngine_t e, CUdeviceptr data,
	int count, int k, selectType_t type, void* value, int* index) {

	CUresult result;
	int numWarps = SetBlockRanges(e, count);
	int blockCount = DivUp(numWarps, e->warpsPerBlock);
	
	CuCallStack callStack;
	callStack.Push(data, e->countMem, e->rangeMem, 0, 6);
	result = e->countFuncs[(int)type]->Launch(blockCount, 1, callStack);
	
	// DEBUG
	std::vector<uint> counts;
	e->countMem->ToHost(counts);
	int hist2[64] = { 0 };
	for(int i(0); i < blockCount * e->warpsPerBlock; ++i)
		for(int j(0); j < 64; ++j)
			hist2[j] += counts[i * 64 + j];

	int histScan[64] = { 0 };
	for(int i(1); i < 64; ++i)
		histScan[i] = hist2[i - 1] + histScan[i - 1];
	// DEBUG


	callStack.Reset();
	callStack.Push(e->countMem, e->scanTotalMem, e->scanWarpMem, numWarps, k);
	result = e->histFuncs[0]->Launch(1, 1, callStack);

	std::vector<uint> globalScan, warpScan;
	e->scanTotalMem->ToHost(globalScan);
	e->scanWarpMem->ToHost(warpScan);

	int b1 = globalScan[128];


	// Allocate space to hold the first level select.
	DeviceMemPtr buffer;
	result = e->context->MemAlloc<uint>(globalScan[b1], &buffer);

	
	// Call the streaming function.




	return SELECT_STATUS_SUCCESS;
}

