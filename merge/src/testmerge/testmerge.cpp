#define NOMINMAX
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <random>
#include <intrin.h>
#include <algorithm>
#include <map>

#include "../../../util/cucpp.h"

#include <vector>

typedef unsigned int uint;
typedef __int64 int64;
typedef unsigned __int64 uint64;

std::tr1::mt19937 mt19937;

const int NumValues = 1024;

template<typename T, typename Pr = std::less<T> > 
class Foo {
public:
	Foo(const Pr& pred = Pr()) : _pred(pred) { }
private:
	const Pr& _pred;
};


int main(int argc, char** argv) {
	Foo<int> f;
/*
	int numThreads = 256;
	int vt = 4;
	int warpSize = 32;
	int numValues = numThreads * vt;
	
	int width = 4 * warpSize;
	int spacing = numValues / width;

	printf("Gather every %dth thread value 0:\n", spacing);
	for(int i(0); i < width; ++i) {
		int index = spacing * i + 0;
		int hash = index + index / warpSize;
		printf("%d: %d (%d - %d)\n", i, index, hash, hash % warpSize);
	}
*/

	for(int tid(0); tid < 32; ++tid) {
		uint index = tid + 9;
		uint i2 = index + index / 32;
		printf("%d: %d %d (%d)\n", tid, index, i2, 31 & i2);

	}
	return 0;






	cuInit(0);

	DevicePtr device;
	ContextPtr context;

	CUresult result = CreateCuDevice(0, &device);
	result = CreateCuContext(device, 0, &context);

	ModulePtr module;
	result = context->LoadModuleFilename("../../src/cubin/merge.cubin", &module);

	// Allocate values.
	std::tr1::uniform_int<int> r1(0, 50);
	
	std::tr1::uniform_int<int> r2(10, 100);

	std::vector<int> valuesHostA(NumValues), valuesHostB(NumValues);
	for(int i(0); i < NumValues; ++i) {
		valuesHostA[i] = r1(mt19937);
		valuesHostB[i] = r2(mt19937);
	}
	std::sort(valuesHostA.begin(), valuesHostA.end());
	std::sort(valuesHostB.begin(), valuesHostB.end());

	DeviceMemPtr valuesDeviceA, valuesDeviceB;
	result = context->MemAlloc(valuesHostA, &valuesDeviceA);
	result = context->MemAlloc(valuesHostB, &valuesDeviceB);

	DeviceMemPtr indicesDevice;
	result = context->MemAlloc<int2>(NumValues, &indicesDevice);

	FunctionPtr func;
	module->GetFunction("SearchBlockInt", make_int3(256, 1, 1), &func);

	CuCallStack callStack;
	callStack.Push(valuesDeviceA, make_int2(0, NumValues),
		valuesDeviceB, make_int2(0, NumValues), indicesDevice);

	func->Launch(1, 1, callStack);

	std::vector<int2> indicesHost;
	indicesDevice->ToHost(indicesHost);

	/*
	for(int i(0); i < 64; ++i) {
		int index1 = std::upper_bound(valuesHostA.begin(), valuesHostA.end(), i) 
			 - valuesHostA.begin();

		int index2 = std::lower_bound(valuesHostB.begin(), valuesHostB.end(), i)
			- valuesHostB.begin();

	//	assert(index1 == indicesHost[index2]);
		printf("%d: %d %d\n", i, index1, indicesHost[index2]);
	}
*/
	int j = 0;
/*
int main(int argc, char** argv) {
	cuInit(0);

	DevicePtr device;
	ContextPtr context;

	CUresult result = CreateCuDevice(0, &device);
	result = CreateCuContext(device, 0, &context);

	ModulePtr module;
	result = context->LoadModuleFilename("../../src/cubin/ranges.cubin", &module);

	// Allocate values.
	std::tr1::uniform_int<int> r1(0, NumActiveValues - 1);

	int NumValues = 1024;
	std::vector<int> valuesHost(NumValues);
	for(int i(0); i < NumValues; ++i) 
		valuesHost[i] = ActiveValues[r1(mt19937)];

	std::sort(valuesHost.begin(), valuesHost.end());

	DeviceMemPtr valuesDevice, rangesDevice;
	result = context->MemAlloc(valuesHost, &valuesDevice);
	context->MemAlloc<int2>(NumRadixSlots, &rangesDevice);

	std::vector<int2> hostRanges(NumRadixSlots);
	std::fill(hostRanges.begin(), hostRanges.end(), make_int2(-1, 0));
	for(int i(0); i < NumValues; ++i) {
		int d = valuesHost[i];
		hostRanges[d].x = std::min(hostRanges[d].x, i);
		hostRanges[d].y = i + 1;
	}
		


	FunctionPtr func;
	module->GetFunction("TestRanges64", make_int3(1024, 1, 1), &func);

	CuCallStack callStack;
	callStack.Push(valuesDevice, rangesDevice);

	func->Launch(1, 1, callStack);

	std::vector<int2> rangesHost;
	rangesDevice->ToHost(rangesHost);

	int j = 0;
*/

	/*
	const int Count = 128;
	std::vector<uint> aHost(Count), bHost(Count);
	int aNext = 1, bNext = 1;
	std::tr1::uniform_int<int> r(0, 6);
	for(int i(0); i < Count; ++i) {
		aHost[i] = aNext;
		bHost[i] = bNext ;
		if(aNext == 10) aNext = 20;
		if(0 == r(mt19937)) ++aNext;
		if(0 == r(mt19937)) ++bNext;
	}

	DeviceMemPtr aDevice, bDevice;
	result = context->MemAlloc(aHost, &aDevice);
	result = context->MemAlloc(bHost, &bDevice);

	DeviceMemPtr cDevice;
	result = context->MemAlloc<uint>(2 * Count, &cDevice);
	
	int2 aRange = make_int2(0, Count);
	int2 bRange = make_int2(0, Count);
	
	ModulePtr module;
	result = context->LoadModuleFilename("hist.cubin", &module);
	
	FunctionPtr func;
	result = module->GetFunction("Hist", make_int3(128, 1, 1), &func);

	DeviceMemPtr debugDevice;
	context->MemAlloc<uint>(16 * Count, &debugDevice);

	CuCallStack cs;
	cs.Push(aDevice, bDevice, aRange, bRange, cDevice, debugDevice);
	result = func->Launch(1, 1, cs);

	std::vector<uint> debugHost;
	result = debugDevice->ToHost(debugHost);

	// Get each start index.
	std::vector<int> starts(32), end(32);
	for(int i(0); i < Count; ++i) {
		if(!i || (aHost[i] != aHost[i - 1]))
			starts[aHost[i]] = i;
		if(i && (aHost[i] != aHost[i - 1]))
			end[aHost[i - 1]] = i;
	}

	for(int i(0); i < 32; ++i) {
		uint packed = debugHost[i];
		int begin = 0xffff & packed;
		int end = packed>> 16;

		printf("-- %2d -- (%d, %d):\n", i, begin, end);
		for(int i(begin); i < std::min(end, Count); ++i)
			printf("\t%d: %d\n", i, aHost[i]);

	}

	int j = 0;
	int i = 0;*/
	
}

