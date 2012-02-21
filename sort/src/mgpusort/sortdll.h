#pragma once

#include "../../../util/cucpp.h"
#include "../../../inc/mgpusort.h"

const int MAX_BITS = 7;

struct sortEngine_d : public CuBase {
	struct CountKernel {
		ModulePtr module;
		FunctionPtr func[MAX_BITS];
	};

	struct HistKernel {
		ModulePtr module;
		FunctionPtr func[MAX_BITS];
	};

	struct DownsweepKernel {
		ModulePtr module;
		FunctionPtr func[MAX_BITS];
	};

	// Sort kernels have a large number of permutations. Load these on demand 
	// after encountering a sort request.
	struct SortKernel {
		ModulePtr module;
		CUtexref keysTexRef;		// "keys_global_in"
		FunctionPtr func[MAX_BITS];
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

	// 2 bytes per digit per sort block.
	DeviceMemPtr countBuffer;

	// 4 bytes per digit per task.
	DeviceMemPtr taskOffsets;

	// 4 bytes per digit.
	DeviceMemPtr digitTotalsScan;

	// 4 bytes per digit per sort block.
	DeviceMemPtr scatterOffsets;

	// As particular kernels are required, load the modules and functions and
	// keep them here.
	std::auto_ptr<CountKernel> count;
	
	std::auto_ptr<HistKernel> hist;

	std::auto_ptr<DownsweepKernel> downsweep;

	// Index 0: block size - 64 or 128 threads.
	// Index 1: values per thread - 16 or 24.
	// Index 2: value counts.
	//		0 - sort keys only
	//		1 - sort keys and emit indices (for first pass)
	//		2 - sort keys and single values
	//		3 - sort keys and multiple values
	std::auto_ptr<SortKernel> sort[2][2][4];
};

typedef intrusive_ptr2<sortEngine_d> EnginePtr;

struct SortTable { 
	char pass[7]; 
	int numSortThreads;
	int valuesPerThread;
	bool useTransList;
};

typedef const char PassTable[6];

SortTable GetOptimizedSortTable(sortData_t data);

