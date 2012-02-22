#include "benchmark.h"
#include "../B40C/b40c/util/ping_pong_storage.cuh"
#include "../B40C/b40c/radix_sort/enactor.cuh"


////////////////////////////////////////////////////////////////////////////////
//

template<typename Storage, typename Enactor>
cudaError_t B40cSortStorage(Storage& s, Enactor& e, int numBits, 
	int numElements) {
	const b40c::radix_sort::ProbSizeGenre SIZE = b40c::radix_sort::UNKNOWN_SIZE;
	switch(numBits) {
		case 1: return e.template Sort<0, 1, SIZE>(s, numElements);
		case 2: return e.template Sort<0, 2, SIZE>(s, numElements);
		case 3: return e.template Sort<0, 3, SIZE>(s, numElements);
		case 4: return e.template Sort<0, 4, SIZE>(s, numElements);

		case 5: return e.template Sort<0, 5, SIZE>(s, numElements);
		case 6: return e.template Sort<0, 6, SIZE>(s, numElements);
		case 7: return e.template Sort<0, 7, SIZE>(s, numElements);
		case 8: return e.template Sort<0, 8, SIZE>(s, numElements);

		case 9: return e.template Sort<0, 9, SIZE>(s, numElements);
		case 10: return e.template Sort<0, 10, SIZE>(s, numElements);
		case 11: return e.template Sort<0, 11, SIZE>(s, numElements);
		case 12: return e.template Sort<0, 12, SIZE>(s, numElements);

		case 13: return e.template Sort<0, 13, SIZE>(s, numElements);
		case 14: return e.template Sort<0, 14, SIZE>(s, numElements);
		case 15: return e.template Sort<0, 15, SIZE>(s, numElements);
		case 16: return e.template Sort<0, 16, SIZE>(s, numElements);

		case 17: return e.template Sort<0, 17, SIZE>(s, numElements);
		case 18: return e.template Sort<0, 18, SIZE>(s, numElements);
		case 19: return e.template Sort<0, 19, SIZE>(s, numElements);
		case 20: return e.template Sort<0, 20, SIZE>(s, numElements);

		case 21: return e.template Sort<0, 21, SIZE>(s, numElements);
		case 22: return e.template Sort<0, 22, SIZE>(s, numElements);
		case 23: return e.template Sort<0, 23, SIZE>(s, numElements);
		case 24: return e.template Sort<0, 24, SIZE>(s, numElements);

		case 25: return e.template Sort<0, 25, SIZE>(s, numElements);
		case 26: return e.template Sort<0, 26, SIZE>(s, numElements);
		case 27: return e.template Sort<0, 27, SIZE>(s, numElements);
		case 28: return e.template Sort<0, 28, SIZE>(s, numElements);

		case 29: return e.template Sort<0, 29, SIZE>(s, numElements);
		case 30: return e.template Sort<0, 30, SIZE>(s, numElements);
		case 31: return e.template Sort<0, 31, SIZE>(s, numElements);
		case 32: return e.template Sort<0, 32, SIZE>(s, numElements);
	} 

	return cudaErrorInvalidValue;
}


////////////////////////////////////////////////////////////////////////////////
// B40cSort

CUresult SetArraysToStorage(b40c::util::PingPongStorage<uint>& storage,
	B40cTerms& terms) {
	CUresult result = terms.randomKeys->ToDevice(0,
		(CUdeviceptr)storage.d_keys[storage.selector], 
		sizeof(uint) * terms.count);
	return result;
}

CUresult SetArraysToStorage(b40c::util::PingPongStorage<uint, uint>& storage, 
	B40cTerms& terms) {
	CUresult result = terms.randomKeys->ToDevice(0,
		(CUdeviceptr)storage.d_keys[storage.selector], 
		sizeof(uint) * terms.count);
	if(CUDA_SUCCESS != result) return result;

	result = terms.randomVals->ToDevice(0,
		(CUdeviceptr)storage.d_values[storage.selector],
		sizeof(uint) * terms.count);
	return result;
}

template<typename Storage>
cudaError_t RunB40cBenchmark(Storage& storage, B40cTerms& terms,
	double* elapsed) {
	
	b40c::radix_sort::Enactor enactor;
	enactor.ENACTOR_DEBUG = false;
	CuEventTimer timer;

	for(int i(0); i < terms.iterations; ++i) {
		if(!i || terms.reset) {
			timer.Stop();
			CUresult result = SetArraysToStorage(storage, terms);
			if(CUDA_SUCCESS != result) return (cudaError_t)result;			
		}
		timer.Start(false);

		cudaError_t error = B40cSortStorage(storage, enactor, terms.numBits,
			terms.count);
		if(cudaSuccess != error) return error;
	}
	*elapsed = timer.Stop();
	return cudaSuccess;
}


cudaError_t B40cBenchmark(B40cTerms& terms, double* elapsed) {
	cudaError_t error = cudaSuccess;
	if(terms.randomVals) {
		b40c::util::PingPongStorage<uint, uint> storage;
		storage.d_keys[1] = (uint*)terms.sortedKeys->Handle();
		storage.d_values[1] = (uint*)terms.sortedVals->Handle();
		
		DeviceMemPtr keys1, values1;
		terms.context->MemAlloc<uint>(RoundUp(terms.count, 512), &keys1);
		terms.context->MemAlloc<uint>(RoundUp(terms.count, 512), &values1);

		storage.d_keys[0] = (uint*)keys1->Handle();
		storage.d_values[0] = (uint*)values1->Handle();

		error = RunB40cBenchmark(storage, terms, elapsed);

		if(!storage.selector) {
			// copy from slot 0 to slot 1
			keys1->ToDevice(0, terms.sortedKeys, 0, 
				sizeof(uint) * terms.count);
			values1->ToDevice(0, terms.sortedVals, 0, 
				sizeof(uint) * terms.count);
		}
	} else {
		b40c::util::PingPongStorage<uint> storage;
		storage.d_keys[1] = (uint*)terms.sortedKeys->Handle();
		
		DeviceMemPtr keys1;
		terms.context->MemAlloc<uint>(RoundUp(terms.count, 512), &keys1);

		storage.d_keys[0] = (uint*)keys1->Handle();

		error = RunB40cBenchmark(storage, terms, elapsed);

		if(!storage.selector)
			// copy from slot 0 to slot 1
			keys1->ToDevice(0, terms.sortedKeys, 0, sizeof(uint) * terms.count);
	}

	return error;
}



