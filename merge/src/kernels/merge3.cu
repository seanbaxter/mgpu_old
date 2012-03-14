#include "ranges.cu"

template<typename T>
DEVICE2 voidb 

template<typename T>
DEVICE2 void SearchBlockConstricted(uint tid, int numThreads, int numValues, 
	const T* aData_shared, uint aCount, const T* bData_shared, uint bCount, 
	uint* indices_shared, int kind, uint* debug_global) {

	// Run a binary search for every Spacing'th element. For 256 threads and 4
	// values per thread (NumValues = 1024), this processes one out of every 32
	// keys. This is only 3% of the total (if bCount is full). This is the most
	// inefficient operation, because each thread does a binary search over all
	// the shared memory values.
	const int Spacing = numValues / WARP_SIZE;
	const int Spacing2 = numThreads / 32;

	if(tid < WARP_SIZE) {
		// NOTE: is there any performance hit in each lane in the warp setting
		// the same shmem element?
		const uint last = Spacing2 * WARP_SIZE;
		indices_shared[last + last / WARP_SIZE] = aCount;
		
		// PROCESS 32 THREADS. (Spacing * tid)

		// Run elements Spacing * tid. For NumValues = 1024, these are elements:
		// 0, 32, 64, 96, 128, etc. 

		// Use strided indexing to retrieve every Spacing'th element without
		// bank conflicts.
		int i = Spacing * tid;
		if(i >= bCount) i = bCount - 1;
		i += i / WARP_SIZE;
	
		T key = bData_shared[i];

		int itCount;
		uint index = RangeBinarySearch(aData_shared, 0, aCount, key, kind, 
			itCount);

		int j = Spacing2 * tid;
		j += j / WARP_SIZE;
		indices_shared[j] = index;

		// PROCESS 32 THREADS (64 done). (Spacing * tid + Spacing / 2)

		// Run elements Spacing * tid + Spacing / 2. We use the indices to the
		// left and right (i.e. Spacing * tid and Spacing * tid + Spacing) to
		// constrain this search. For NumValues = 1024, these are elements:
		// 16, 48, 80, 112, 144, etc.
	
		j = Spacing2 * tid;
		j += j / WARP_SIZE;
		int j2 = Spacing2 * tid + Spacing2;
		j2 += j2 / WARP_SIZE;

		uint begin = indices_shared[j];
		uint end = indices_shared[j2];

		i = Spacing * tid + Spacing / 2;
		i += i / WARP_SIZE;

		key = bData_shared[i];
		index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			itCount); 
	
		indices_shared[j + Spacing2 / 2] = index;
	}
	__syncthreads();

	// PROCESS 64 THREADS (128 done).

	// Run elements (Spacing / 2) * tid + Spacing / 4. Constrain these searches
	// with the results of the previous two runs. For NumValues = 1024, these
	// are elements:
	// 8, 40, 56, 72, 88, 104, etc.
	if(numThreads >= 128) && (tid < 2 * WARP_SIZE)) {
		const int Spacing3 = Spacing2 / 2;
		int j = Spacing3 * tid;
		j += j / WARP_SIZE;
		int j2 = Spacing3 * tid + Spacing3;
		j2 += j2 / WARP_SIZE;

		uint begin = indices_shared[j];
		uint end = indices_shared[j2];
		
		int i = (Spacing / 2) * tid + Spacing / 4;
		i += i / WARP_SIZE;

		T key = bData_shared[i];
		int itCount;
		uint index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			itCount);
	
		indices_shared[j + Spacing3 / 2] = index;
	}
	__syncthreads();

	// PROCESS 128 THREADS (256 done). Only do this for thread blocks of size 
	// 256.
	if((256 == numThreads) && (tid < 4 * WARP_SIZE)) {
		const int Spacing4 = Spacing2 / 4;
		int j = Spacing4 * tid;
		j += j / WARP_SIZE;
		int j2 = Spacing4 * tid + Spacing4;
		j2 += j2 / WARP_SIZE;

		uint begin = indices_shared[j];
		uint end = indices_shared[j2];
		
		int i = (Spacing / 4) * tid + Spacing / 8;
		i += i / WARP_SIZE;

		T key = bData_shared[i];
		int itCount;
		uint index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			itCount);
	
		indices_shared[j + Spacing4 / 2] = index;
	}
	__syncthreads();
}

template<typename T>
DEVICE2 void SearchThreadRecursive(uint tid, const T* aData_shared, uint bCount,
	uint begin, uint end, const T* keys, int i, int delta, uint* indices,
	uint* target_shared, int kind, int store, uint* debug_global) {

	// Run the binary search at keys[i].
	int itCount;
	uint index = RangeBinarySearch(aData_shared, begin, end, keys[i], kind, 
		itCount);

	if(0 == store) {
		// For a lower_bound/upper_bound search, store the indices in thread
		// order (strided).
		target_shared[i] = index;
	} else {
		// Scatter for merge.
	}

	// Recurse to the left and right if delta > 0.
	if(delta > 0) {
		SearchThreadRecursive(tid, aData_shared, bCount, begin, index, keys, 
			i - delta, delta / 2, indices, target_shared, kind, store, 
			debug_global);

		SearchThreadRecursive(tid, aData_sharde, bCount, index, end, keys, 
			i + delta, delta / 2, indices, target_shared, kind, store,
			debug_global);
	}
}



////////////////////////////////////////////////////////////////////////////////

typedef float T;
const int NumThreads = 256;
const int ValuesPerThread = 8;
const int NumValues = NumThreads * ValuesPerThread;
const int NumWarps = NumThreads / WARP_SIZE;

extern "C" __global__
void Test(const T* a_global, T* b_global, uint* indices_global) {
	__shared__ T a_shared[NumValues];
	__shared__ T b_shared[NumValues + NumValues / WARP_SIZE];

	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;

	#pragma unroll
	for(int i = 0; i < ValuesPerThread; ++i) {
		uint index = tid + i * NumThreads;
		a_shared[index] = a_global[index];
		b_shared[index + index / WARP_SIZE] = b_global[index];
	}
	__syncthreads();

	__shared__ uint indices_shared[NumValues];

	SearchBlockConstricted(tid, NumThreads, NumValues, a_shared, NumValues,
		b_shared, NumValues, indices_shared, 0, 0);

	#pragma unroll
	for(int i = 0; i < ValuesPerThread; ++i) {
		uint index = tid + i * NumThreads;
		indices_global[index] = indices_shared[index];
	}
}

