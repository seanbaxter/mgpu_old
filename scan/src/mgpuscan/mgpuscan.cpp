#include "../../../inc/mgpuscan.h"
#include "../../../util/cucpp.h"
#include <sstream>



// TODO: fiddle with these. probably put up to 64 threads and have 33% 
// occupancy.
const int NumScanThreads = 256;
const int ScanValuesPerThread = 8;
const int ScanValuesPerBlock = ScanValuesPerThread * NumScanThreads;

const int NumSegScanThreads = 256;
const int SegScanValuesPerThread = 8;
const int SegScanValuesPerBlock = NumSegScanThreads * SegScanValuesPerThread;


////////////////////////////////////////////////////////////////////////////////

const char* ScanStatusStrings[] = {
	"SCAN_STATUS_SUCCESS"
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

	FunctionPtr scanFuncs[3];
	FunctionPtr scanFlagFuncs[3];
	FunctionPtr scanKeysFuncs[3];

	DeviceMemPtr blockScanMem;
	DeviceMemPtr headFlagsMem;
	DeviceMemPtr rangeMem;
	int numScanBlocks;
	int numSegBlocks;

	std::vector<int2> ranges;
};

struct FuncParams {
	const char* pass1, *pass2, *pass3;
	int numThreads;
};

CUresult LoadFuncs(CuModule* module, FuncParams params, FunctionPtr pass[3]) {

	CUresult result = module->GetFunction(params.pass1, 
		make_int3(params.numThreads, 1, 1), &pass[0]);
	if(CUDA_SUCCESS != result) return result;

	result = module->GetFunction(params.pass2,
		make_int3(params.numThreads, 1, 1), &pass[1]);
	if(CUDA_SUCCESS != result) return result;

	result = module->GetFunction(params.pass3,
		make_int3(params.numThreads, 1, 1), &pass[2]);
	return result;
}



////////////////////////////////////////////////////////////////////////////////


scanStatus_t SCANAPI scanCreateEngine(const char* cubin, scanEngine_t* engine) {
	std::auto_ptr<scanEngine_d> e(new scanEngine_d);
	
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_INVALID_CONTEXT;

	if(2 != e->context->Device()->ComputeCapability().first)
		return SCAN_STATUS_UNSUPPORTED_DEVICE;

	result = e->context->LoadModuleFilename(cubin, &e->module);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_NOT_FOUND;

	FuncParams funcParams;
	funcParams.numThreads = NumScanThreads;
	funcParams.pass1 = "GlobalScanUpsweep";
	funcParams.pass2 = "GlobalScanReduction";
	funcParams.pass3 = "GlobalScanDownsweep";
	result = LoadFuncs(e->module, funcParams, e->scanFuncs);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

	funcParams.numThreads = NumSegScanThreads;
	funcParams.pass1 = "SegScanUpsweepFlag";
	funcParams.pass2 = "SegScanReduction";
	funcParams.pass3 = "SegScanDownsweepFlag";
	result = LoadFuncs(e->module, funcParams, e->scanFlagFuncs);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

	funcParams.pass1 = "SegScanUpsweepKeys";
	funcParams.pass2 = "SegScanReduction";
	funcParams.pass3 = "SegScanDownsweepKeys";
	result = LoadFuncs(e->module, funcParams, e->scanKeysFuncs);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_KERNEL_ERROR;

	int numSMs = e->context->Device()->Attributes().multiprocessorCount;

	// Launch 4 256-thread blocks per SM.
	e->numScanBlocks = 4 * numSMs;
	e->numSegBlocks = 4 * numSMs;

	int numBlocks = std::max(e->numScanBlocks, e->numSegBlocks);

	// Allocate a uint per thread block plus a uint for the scan total.
	result = e->context->MemAlloc<uint>(numBlocks + 1, &e->blockScanMem);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_DEVICE_ALLOC_FAILED;

	result = e->context->MemAlloc<uint2>(numBlocks, &e->rangeMem);
	result = e->context->MemAlloc<uint>(e->numSegBlocks, &e->headFlagsMem);
	
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
// 

int SetBlockRanges(scanEngine_t engine, int count, bool segScan) {

	int blockSize = segScan ? SegScanValuesPerBlock : ScanValuesPerBlock;
	int numBlocks = segScan ? engine->numSegBlocks : engine->numScanBlocks;
	
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
		range.y = std::min(range.x + bricks * 256 * 8, count);
		engine->ranges[i] = range;
	}
	engine->rangeMem->FromHost(engine->ranges);
	
	return (int)engine->ranges.size();
}


scanStatus_t SCANAPI scanArray(scanEngine_t engine, CUdeviceptr values,
	CUdeviceptr scan, int count, uint init, uint* scanTotal, bool inclusive) {

	if(!engine) return SCAN_STATUS_INVALID_VALUE;

	int numBlocks = SetBlockRanges(engine, count, false);

	// Don't run the upsweep on the last block - it contributes nothing to
	// reduction.
	CuCallStack callStack;
	callStack.Push(values, engine->rangeMem, engine->blockScanMem);
	CUresult result = engine->scanFuncs[0]->Launch(numBlocks - 1, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

	// Run a reduction for the block offsets if numBlocks > 1. We've already 
	// poked a 0 to the starto f blockScanMem, so we can pass this step in the
	// case of a single block scan.
	if(numBlocks > 1) {
		callStack.Reset();
		callStack.Push(engine->blockScanMem, numBlocks);
		result = engine->scanFuncs[1]->Launch(1, 1, callStack);
		if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;
	}

	// Retrieve the scan total from the end of blockScanMem.
	if(scanTotal)
		engine->blockScanMem->ToHostByte(scanTotal, sizeof(uint) * numBlocks,
			sizeof(uint));

	callStack.Reset();
	callStack.Push(values, scan, engine->blockScanMem, engine->rangeMem, 
		count, init, (int)inclusive);
	result = engine->scanFuncs[2]->Launch(numBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

	return SCAN_STATUS_SUCCESS;
}


scanStatus_t SCANAPI scanSegmentedFlag(scanEngine_t engine, CUdeviceptr packed,
	CUdeviceptr scan, int count, uint init, bool inclusive) {

	if(!engine) return SCAN_STATUS_INVALID_VALUE;

	int numBlocks = SetBlockRanges(engine, count, true);

	CuCallStack callStack;
	CUresult result;

	callStack.Push(packed, engine->blockScanMem, engine->headFlagsMem,
		engine->rangeMem);
		
	result = engine->scanFlagFuncs[0]->Launch(numBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

//	std::vector<uint> blockTotalsHost;
//	engine->blockScanMem->ToHost(blockTotalsHost);

//	std::vector<uint> headFlagsHost;
//	engine->headFlagsMem->ToHost(headFlagsHost);

	callStack.Reset();
	callStack.Push(engine->headFlagsMem, engine->blockScanMem, numBlocks);
	result = engine->scanFlagFuncs[1]->Launch(1, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

//j	std::vector<uint> blockScanHost;
//	engine->blockScanMem->ToHost(blockScanHost);

	callStack.Reset();
	callStack.Push(packed, scan, engine->blockScanMem, engine->rangeMem, 
		count, init, (int)inclusive);
	result = engine->scanFlagFuncs[2]->Launch(numBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SCAN_STATUS_LAUNCH_ERROR;

	return SCAN_STATUS_SUCCESS;

}

scanStatus_t SCANAPI scanSegmentedKeys(scanEngine_t engine, CUdeviceptr values,
	CUdeviceptr keys, CUdeviceptr scan, int count, uint init, bool inclusive) {


	return SCAN_STATUS_SUCCESS;
}