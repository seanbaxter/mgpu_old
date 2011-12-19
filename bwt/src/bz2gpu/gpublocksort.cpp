// (c) 2011 Sean Baxter (www.moderngpu.com)

// This file drops in for bwt/bzip2-1.0.6/blocksort.c to enable CUDA BWT block
// sorting. blocksort.c must be excluded from the build to prevent link issues.

extern "C" {
#include "../../bzip2-1.0.6/bzlib_private.h"
}

#include "../../../inc/mgpubwt.h"
#include "../../../util/cucpp.h"
#include <memory>
#include <cstdio>

#ifdef WIN32
#include <windows.h>
#endif

const char* CubinPath = "cubin/";

// bz2 blocksort runs in-place. It provides auxiliary arrays that are unneeded
// here (we maintain our own in the bwtEngine_t type).
/* Pre:
      nblock > 0
      arr2 exists for [0 .. nblock-1 +N_OVERSHOOT]
      ((UChar*)arr2)  [0 .. nblock-1] holds block
      arr1 exists for [0 .. nblock-1]

   Post:
      ((UChar*)arr2) [0 .. nblock-1] holds block
      All other areas of block destroyed
      ftab [ 0 .. 65536 ] destroyed
      arr1 [0 .. nblock-1] holds sorted order
*/

struct CudaSupport {
	DevicePtr device;
	ContextPtr context;
	bwtEngine_t engine;
};

bool CreateCudaSupport(std::auto_ptr<CudaSupport>* cudaSupport) {
	CUresult result = cuInit(0);
	if(CUDA_SUCCESS != result) {
		fprintf(stderr, "Could not initialize CUDA library.\n");
		return false;
	}

	std::auto_ptr<CudaSupport> c(new CudaSupport);
	result = CreateCuDevice(0, &c->device);
	if(CUDA_SUCCESS != result) {
		fprintf(stderr, "Could not create CUDA driver API device.\n");
		return false;		
	}
	if(2 != c->device->ComputeCapability().first) {
		fprintf(stderr, "CUDA device must have 2.x compute capability.\n");
		return false;
	}

	result = CreateCuContext(c->device, 0, &c->context);
	if(CUDA_SUCCESS != result) {
		fprintf(stderr, "Could not create CUDA context.\n");
		return false;
	}

	bwtStatus_t status = bwtCreateEngine(CubinPath, &c->engine);
	if(BWT_STATUS_SUCCESS != status) {
		fprintf(stderr, "Could not load MGPU-BWT: %s.\n", 
			bwtStatusString(status));
		return false;
	}

	*cudaSupport = c;
	return true;
}

extern "C" void BZ2_blockSort(EState* s) {
	int count = s->nblock;
	const byte* symbols = (byte*)s->arr2;
	int* indices = (int*)s->arr1;

	static CudaSupport* cudaSupport = 0;
	
	if(!cudaSupport) {
		if(s->verbosity >= 4) 
			printf("Creating CUDA device and loading MGPU-BWT library...\n");
		std::auto_ptr<CudaSupport> c;
		bool success = CreateCudaSupport(&c);
		if(!success) exit(1);

		cudaSupport = c.release();
	}

#if defined(WIN32) && defined(BENCHMARK)
	LARGE_INTEGER freq, begin, end;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&begin);
#endif

	// NOTE: only retrieve the indices, not the re-ordered symbols. The bz2
	// code does the string permutation in compress.c: generateMTFValues.
	int segCount;
	float avSegSize;
	bwtStatus_t status = bwtSortBlock(cudaSupport->engine, symbols, count, 16,
		0, indices, &segCount, &avSegSize);
	if(BWT_STATUS_SUCCESS != status) {
		fprintf(stderr, "Failure in MGPU-BWT block sort: %s.\n",
			bwtStatusString(status));
		exit(0);
	}
#if defined(WIN32) && defined(BENCHMARK)
	QueryPerformanceCounter(&end);
	int64 diff = end.QuadPart - begin.QuadPart;
	double elapsed = (double)diff / freq.QuadPart;

	static double totalElapsed = 0;
	totalElapsed += elapsed;
	printf("%6d %9.5f %10.6f\n", segCount, avSegSize, totalElapsed);
#endif

	// Advance origPtr to the index i of s->arr1[i] which holds 0. This is the 
	// "start" of the original array. Not sure why bz2 code puts it here instead
	// of somewhere else becaues it's not a block sorting operation.
	s->origPtr = -1;
	for (int i = 0; i < s->nblock; i++)
		if (s->ptr[i] == 0)
			{ s->origPtr = i; break; };
}

