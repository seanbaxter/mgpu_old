#include <vector>
#include <memory>
#include <cstdio>
#include "../../../util/cucpp.h"

#ifdef _MSC_VER
#include <random>
#else
#include <tr1/random>
#endif

std::tr1::mt19937 mt19937;


struct MaxIndexEngine {
	ContextPtr context;
	ModulePtr module;
	FunctionPtr pass1, pass2;
	DeviceMemPtr maxMem, indexMem;
	DeviceMemPtr rangeMem;
	int numBlocks;
};

CUresult CreateMaxIndexEngine(const char* cubin, 
	std::auto_ptr<MaxIndexEngine>* ppEngine) {

	std::auto_ptr<MaxIndexEngine> e(new MaxIndexEngine);
		
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return result;

	if(2 != e->context->Device()->ComputeCapability().first)
	return CUDA_ERROR_INVALID_DEVICE ;

	result = e->context->LoadModuleFilename(cubin, &e->module);
	if(CUDA_SUCCESS != result) return result;

	result = e->module->GetFunction("FindMaxIndexLoop", make_int3(256, 1, 1),
		&e->pass1);
	if(CUDA_SUCCESS != result) return result;

	result = e->module->GetFunction("FindMaxIndexReduce", make_int3(256, 1, 1),
		&e->pass2);
	if(CUDA_SUCCESS != result) return result;

	int numSMs = e->context->Device()->Attributes().multiprocessorCount;

	// Launch 6 256-thread blocks per SM.
	e->numBlocks = 6 * numSMs;

	// Allocate an element for each thread block.
	result = e->context->MemAlloc<float>(e->numBlocks, &e->maxMem);
	if(CUDA_SUCCESS != result) return result;
	result = e->context->MemAlloc<uint>(e->numBlocks, &e->indexMem);
	if(CUDA_SUCCESS != result) return result;

	result = e->context->MemAlloc<uint2>(e->numBlocks, &e->rangeMem);

	*ppEngine = e;
	return CUDA_SUCCESS;

}

CUresult FindGlobalMax(MaxIndexEngine* engine, CUdeviceptr data, int count,
	float* maxX, int* maxIndex) {
	
	// Process 256 values a time .
	int numBricks = DivUp(count, 256);
	int numBlocks = std::min(numBricks, engine->numBlocks);

	// Distribute the work along complete bricks.
	div_t brickDiv = div(numBricks, numBlocks);
	std::vector<int2> ranges(numBlocks);
	for(int i(0); i < numBlocks; ++i) {
		int2 range;
		range.x = i ? ranges[i - 1].y : 0;
		int bricks = (i < brickDiv.rem) ? (brickDiv.quot + 1) : brickDiv.quot;
		range.y = std::min(range.x + bricks * 256, count);
		ranges[i] = range;
	}
	engine->rangeMem->FromHost(ranges);

	CuCallStack callStack;
	callStack.Push(data, engine->maxMem, engine->indexMem, engine->rangeMem);
	CUresult result = engine->pass1->Launch(numBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return result;

	callStack.Reset();
	callStack.Push(engine->maxMem, engine->indexMem, numBlocks);
	result = engine->pass2->Launch(1, 1, callStack);
	if(CUDA_SUCCESS != result) return result;

	// Retrieve the max elements.
	engine->maxMem->ToHost(maxX, 1);
	engine->indexMem->ToHost(maxIndex, 1);
	return CUDA_SUCCESS;
}


int main(int argc, char** argv) {
	cuInit(0);
	
	DevicePtr device;
	CUresult result = CreateCuDevice(0, &device);

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);

	std::auto_ptr<MaxIndexEngine> engine;
	result = CreateMaxIndexEngine("../../src/maxindex/maxindex.cubin", &engine);
	if(CUDA_SUCCESS != result) {
		printf("Could not create max index engine.\n");
		return 0;
	}

	// Search through 5 million elements.
	const int NumElements = 5000000;
	std::vector<float> data(NumElements);
	std::tr1::uniform_real<float> r(-1e9, 1e9);
	for(int i(0); i < NumElements; ++i)
		data[i] = r(mt19937);
	
	// Use CPU to find the max element and index.
	float maxX = -1e37f;
	int maxIndex = 0;

	for(int i(0); i < NumElements; ++i)
		if(data[i] > maxX) {
			maxX = data[i];
			maxIndex = i;
		}

	printf("CPU says max x = %f, max index = %d.\n", maxX, maxIndex);

	// Use GPU to find the max element and index.
	DeviceMemPtr deviceData;
	context->MemAlloc(data, &deviceData);

	result = FindGlobalMax(engine.get(), deviceData->Handle(), NumElements, 
		&maxX, &maxIndex);
	if(CUDA_SUCCESS != result) {
		printf("Failure running max index kernel.\n");
		return 0;
	}

	printf("GPU says max x = %f, max index = %d.\n", maxX, maxIndex);
}

