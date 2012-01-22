#pragma once

#include "../../../util/cucpp.h"
#include "../../../inc/mgpusort.h"


struct sortEngine_d : public CuBase {
	// A single count kernel is used for all sorts. The sort block size is 
	// dynamically specified by a kernel argument.
	struct CountKernel {
		ModulePtr module;
		FunctionPtr functions[6];
		FunctionPtr eeFunctions[6];
	};

	// Histogram kernels can be parameterized over simple scatter or transaction
	// list scatter. Histogram kernel is independent of sortNumThreads or
	// valuesPerThread.
	struct HistKernel {
		ModulePtr module;
		FunctionPtr pass1[6], pass2[6], pass3[6];
	};

	// Sort kernels have a large number of permutations. Load these on demand 
	// after encountering a sort request.
	struct SortKernel {
		ModulePtr module;
		FunctionPtr functions[6];
		FunctionPtr eeFunctions[6];
	};

	// Context that was current when the sort engine was created. This pointer
	// is merely 'attached', so it won't destroy the context when the engine is
	// destroyed.
	ContextPtr context;
	int numSMs;

	// Path of the .cubin files.
	std::string kernelPath;

	// Holds the fractional part of the last block
	DeviceMemPtr keyRestoreBuffer;
	int restoreSourceSize;
	int restoreTargetOffset;

	// Holds temporary buffers. These can be resized prior to sort.
	// Temporaries for count pass.
	DeviceMemPtr countBuffer;
	GlobalMemPtr sortDetectCounters;
	
	// Temporaries for hist pass.
	DeviceMemPtr columnScan, countScan, bucketCodes;

	// As particular kernels are required, load the modules and functions and
	// keep them here.
	std::auto_ptr<CountKernel> count;
	
	// hist[0] is simple scatter, hist[1] is transaction list
	std::auto_ptr<HistKernel> hist[2];

	// The outer index is scatter strategy (simple of transList).
	// The second index is the block size - 128 or 256 threads.
	// The third index is valuesPerThread - 4 or 8.
	// The inner index is the value behavior of the sort kernel:
	//		0 - sort keys only
	//		1 - sort keys and emit indices (for first pass)
	//		2 - sort keys and single values
	//		3 - sort keys and multiple values
	std::auto_ptr<SortKernel> sort[2][2][2][4];
};

typedef intrusive_ptr2<sortEngine_d> EnginePtr;

struct SortTable { 
	char pass[6]; 
	int numSortThreads;
	int valuesPerThread;
	bool useTransList;
};

typedef const char PassTable[6];

SortTable GetOptimizedSortTable(sortData_t data);

