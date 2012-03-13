#include "ranges.cu"

////////////////////////////////////////////////////////////////////////////////

/*
template<typename T, int Count, int Capacity>
struct CircularBuffer {
	static const int Stride = WARP_SIZE + 1;
	static const int NumWarps = Count / WARP_SIZE;
	static const int NumWarpsCapacity = Capacity / WARP_SIZE;
	static const int StridedCapacity = NumWarpsCapacity * Stride;

	__shared__ T data[StridedCapacity];
	uint i;

	uint 

	T Get(uint index) const {
		index += offset;
		if(index >= Capacity) index -= Capacity;
		uint i2 = index + index / WARP_SIZE;
		return data[i2];		
	}

};
*/

////////////////////////////////////////////////////////////////////////////////
// SearchBlockConstricted

// Finds the search index for the first value in each thread. This is done in
// multiple passes. The first pass of 32 elements searches the entire 
// aData_shared array. The next pass of 32 elements is constrained by the 
// results of the first pass. The third pass of 64 elements is constrained by
// the results of the first two passes. The fourth pass (only for 256 threads
// per block) is constrained by all the preceding passes.

// With 4 values per thread, 25% of the searching is done in this function.
// The remaining 75% is executed in SearchThread4, a generally high-performing
// function due to all searches being well constrained.

// NOTE: prefill indices_shared[numThreads - 1] with aCount.

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
		// PROCESS 32 THREADS. (Spacing * tid)

		// Run elements Spacing * tid. For NumValues = 1024, these are elements:
		// 0, 32, 64, 96, 128, etc. 

		// Use strided indexing to retrieve every Spacing'th element without
		// bank conflicts.
		int i = Spacing * tid;
		if(i >= bCount) i = bCount - 1;
		i += i / WARP_SIZE;
	
		T key = bData_shared[i2];

		int count;
		uint index = RangeBinarySearch(aData_shared, 0, aCount, key, kind, 
			count);

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

		key = bData_shared[i2];
		index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			count);
	
		indices_shared[j + Spacing2 / 2] = index;
	}
	__syncthreads();

	// PROCESS 64 THREADS (128 done).

	// Run elements (Spacing / 2) * tid + Spacing / 4. Constrain these searches
	// with the results of the previous two runs. For NumValues = 1024, these
	// are elemenst:
	// 8, 40, 56, 72, 88, 104, etc.
	if(tid < 2 * WARP_SIZE) {
		const int Spacing3 = Spacing2 / 2;
		int j = Spacing3 * tid;
		j += j / WARP_SIZE;
		int j2 = Spacing3 * tid + Spacing3;
		j2 += j2 / WARP_SIZE;

		uint begin = indices_shared[j];
		uint end = indices_shared[j2];
		
		int i = (Spacing / 2) * tid + Spacing / 4;
		i += i / WARP_SIZE;

		T key = bData_shared[i2];
		uint index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			count);
	
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

		T key = bData_shared[i2];
		uint index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
			count);
	
		indices_shared[j + Spacing4 / 2] = index;
	}
	__syncthreads();
}


////////////////////////////////////////////////////////////////////////////////

// SearchThread4 performs the per-thread searches that were started in 
// SearchBlockConstricted.

// If store is 0: store directly to target_shared with WARP_SIZE stride. This is
// used for lower_bound/upper_bound search.

// If store is 1: scatter to 4 * tid + n to 4 * tid + n + index[n] with no
// stride. This is used for sorted array merge.

template<typename T>
DEVICE2 void SearchThread4(uint tid, const T* aData_shared, uint bCount, 
	const uint* indices_shared, const T keys[4], uint indices[4], 
	uint* target_shared, int store) {

	uint activeThreads = DivUp(bCount, 4u);
	if(tid < activeThreads) {

		// Retrieve the already-computed index for values 0 and 3:
		int i = tid + tid / WARP_SIZE;
		int i2 = tid + 1;
		i2 += i2 / WARP_SIZE;

		uint index0 = indices_shared[i];
		uint index4 = indices_shared[i2];

		int count1, count2, count3;
		uint index2 = RangeBinarySearch(aData_shared, index0, index4, keys[2],
			kind, count2);
		uint index1 = RangeBinarySearch(aData_shared, index0, index2, keys[1],
			kind, count1);
		uint index3 = RangeBinarySearch(aData_shared, index2, index4, keys[3],
			kind, count3);

		indices[0] = index0;
		indices[1] = index1;
		indices[2] = index2;
		indices[3] = index3;

		if(0 == store) {
			i = 4 * tid;
			i += i / WARP_SIZE;
			target_shared[i] = index0;
			target_shared[i + 1] = index1;
			target_shared[i + 2] = index2;
			target_shared[i + 3] = index3;
		} else if(1 == store) {
			target_shared[4 * tid + index0] = 4 * tid;
			target_shared[4 * tid + index1] = 4 * tid + 1;
			target_shared[4 * tid + index2] = 4 * tid + 2;
			target_shared[4 * tid + index3] = 4 * tid + 3;
		}
	}
	__syncthreads();
}


////////////////////////////////////////////////////////////////////////////////
// Compute how many values to consume in the B array. We look for the first 
// element in b that is not <=/< aLast (for lower_bound/upper_bound).

// We use a "btree" ballot search. This is similar to the MGPU lower_bound btree
// search. Because there is a misalignment slot between every set of 32 values,
// we can load values 32 apart into a single warp without bank conflicts. Each 
// lane compares against aLast and ballot is used to find the interval to zoom
// in on. This technique requires two passes, although only one warp is
// involved.

// Only call FindBIntoA with 32 threads!

template<typename T>
DEVICE2 uint FindBIntoA(uint tid, const T* bData_shared, uint bCount, T aLast,
	int kind) {

	uint last = bCount - 1;
	last += last / WARP_SIZE;
	uint index = min(33 * tid + 31, last); 
	T key = bData_shared[index];

	// Find the first key in bData_shared that is not inRange
	bool inRange = kind ? (key < aLast) : (key <= aLast);
	uint bits = __ballot(inRange);

	// clz returns the number of consecutive zero bits starting at the most
	// significant bit. Subtract from 32 to find the warp containing the 
	// first out of range value. Note that clz will not return 0 - we know 
	// there is at least one key (the last one) that is out of range, or 
	// else we wouldn't take this FindAIntoB branch.
	uint warp = 32 - __clz(bits);

	index = min(33 * warp + tid, last);
	key = bData_shared[index];

	// Make the same comparison and share bits using ballot.
	inRange = kind ? (key < aLast) : (key <= aLast);
	bits = __ballot(inRange);

	index = 32 * warp + (32 - __clz(bits));

	return index;
}

////////////////////////////////////////////////////////////////////////////////
// Compute how many values to eat away from the A array. We look for the first
// element in a that is not <=/< bLast (for lower_bound/upper_bound).

// Only call with 32 threads!

template<typename T>
DEVICE2 uint FindAIntoB(uint tid, const T* aData_shared, uint aCount, T bLast,
	uint numValues, int kind) {

	// Data in aData_shared is not strided every 32 elements like bData_shared.
	// Because of this, it's not possible to sample every 32nd element without
	// 32-way bank conflicts.

	// Instead, same every 17th element. If we have 512 or fewer elements, just
	// a single warp (32 * 17 = 544) is enough to search. For 1024 elements, we
	// need to do high and low searches.

	uint index;
	if(numValues <= 512) {
		uint last = aCount - 1;
		index = min(17 * tid + 16, last);
		T key = aData_shared[index];

		// Find the first key in bData_shared that is not inRange.
		bool inRange = kind ? (key <= bLast) : (key < bLast);
		uint bits = __ballot(inRange);

		uint section = 32 - __clz(bits);
		index = min(17 * section + tid, last);
		key = aData_shared[index];

		// Make the same comparison and share bits using ballot.
		inRange = kind ? (key <= bLast) : (key < bLast);
		bits = __ballot(inRange);

		index = 17 * section + (32 - __clz(bits));
	} else if(1024 == numValues) {
		uint last = aCount - 1;
		uint indexLow = min(17 * tid + 16, last);
		uint indexHigh = min(544 + 17 * tid + 16, last);
		T keyLow = aData_shared[indexLow];
		T keyHigh = aData_shared[indexHigh];

		bool inRangeLow = kind ? (keyLow <= bLast) : (keyHigh < bLast);
		bool inRangeHigh = kind ? (keyLow <= bLast) : (keyHigh < bLast);
		uint bitsLow = __ballot(inRangeLow);
		uint bitsHigh = __ballot(inRangeHigh);

		uint countLow = __clz(bitsLow);
		uint countHigh = __clz(bitsHigh);
		uint section = countLow ? (32 - countLow) : (64 - countHigh);

		index = min(17 * section + tid, last);
		T key = aData_shared[index];
		bool inRange = kind ? (key <= bLast) : (key < bLast);
		uint bits = __ballot(inRange);

		index = 17 * section + (32 - __clz(bits));
	}
	return index;
}


////////////////////////////////////////////////////////////////////////////////
// FindStreamConsumed
// Returns the number of A and B array elements to consume this iteration.

template<typename T>
DEVICE2 uint2 FindStreamConsumed(uint tid, const T* aData_shared, uint aCount,
	const T* bData_shared, uint bCount, int kind, uint numValues) {

	__shared__ uint packedCounts_shared;
	if(tid < WARP_SIZE) {
		T aLast = aData_shared[aCount - 1];
		T bLast = bData_shared[bCount - 1];
		bool pred = kind ? (bLast < aLast) : (bLast <= aLast);
		if(pred)
			aCount = FindAIntoB(tid, aData_shared, aCount, bLast, numValues, 
				kind);
		else
			bCount = FindBIntoA(tid, bData_shared, bCount, aLast, kind);
			
		uint packed = bfi(aCount, bCount, 16, 16);
	
		if(!tid) packedCounts_shared = packed;
	}
	__syncthreads();

	uint packed = packedCounts_shared;
	uint2 consumed = make_uint2(0xffff & packed, packed>> 16);
	return consumed;
}


////////////////////////////////////////////////////////////////////////////////
/*
template<int NumThreads, int VT, typename T>
DEVICE2 void SearchBlock(const T* aData_global, int2 aRange,
	const T* bData_global, int2 bRange, int kind, int2* indices_global) { 






		T a = (tid >= aLoaded) ? 
			aData_global[min(aRange.y - 1, aRange.x + tid)] : 
			aData_shared[tid + result.aConsume];
		T b = (tid >= bLoaded) ?
			bData_global[min(bRange.y - 1, bRange.x + tid)] :
			bData_shared[tid + result.bConsume];
		__syncthreads();

		aData_shared[tid] = a;
		bData_shared[tid] = b;
		__syncthreads();









	const int NumValues = NumThreads * VT;
	const int Spacing = WARP_SIZE + 1;
	const int Capacity = NumValues + WARP_SIZE;
	const int BufferSize = Capacity + Capacity / WARP_SIZE;

	__shared__ T aData_shared[NumValues]; 
	__shared__ T bData_shared[BufferSize];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	uint warpValOffset = warp * WARP_SIZE * VT;

	int aLoaded = 0;

	// Round bData_global down to be aligned with the start of a segment.
	bData_global -= RoundDown(bRange.x, WARP_SIZE);

	// bStoreOffset is the offset in bData_shared (circular, strided) to write 
	// the next loaded segment.
	uint bStoreOffset = 0;

	// bLoadOffset is the offset in bData_shared corresponding to the next 
	// loaded value for tid 0.
	uint bLoadOffset = 0;

	// Number of individual elements available for reading starting at
	// bLoadOffset. Note that circular arithmetic is required to find the 
	// actual offsets.
	uint bLoaded = 0;

	// Prime the circular buffer by loading the first segment.
	if(bRange.x < bRange.y) {
		if(tid < WARP_SIZE)
			bData_shared[lane] = bData_global[lane];
		bData_global += WARP_SIZE;

		bStoreOffset = WARP_SIZE;
		bLoadOffset = (WARP_SIZE - 1) & bRange.x;
		bLoaded = WARP_SIZE - bLoadOffset;
	}

	uint aRemaining = aRange.y - aRange.x;
	uint bRemaining = bRange.y - bRange.x;

	InsertResult<VT> result;

	while(aRemaining || bRemaining) {
		
		////////////////////////////////////////////////////////////////////////
		// Load the values to fill the buffers.

		uint aCount = min(NumValues, aRemaining);
		uint bCount = min(NumValues, bRemaining);

		T a = (tid >= aLoaded) ? 
			aData_global[min(aRange.y - 1, aRange.x + tid)] : 
			aData_shared[tid + result.aConsume];

		// We need at least bCount loaded values in the b array yet we have only
		// bLoaded available. Compute the number of segments to pull in to give
		// us what is required.
		uint bLoadCount = RoundUp(bCount - bLoaded, WARP_SIZE);
		LoadSegmentsCircular(tid, NumThreads, bStoreOffset, Capacity,
			bLoadCount, bData_global);
		bData_global += bLoadCount;
		bStoreOffset += bLoadCount;
		if(bStoreOffset > Capacity) 
			bStoreOffset -= Capacity;
		bLoaded += bLoadCount;

		// Read VT elements from the circular array to register.
		T b[VT];

		// If this warp straddles the end of the circular buffer, we need to do
		// explicit range checking. This is expensive, so we try to branch out
		// of it for all the other warps.
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			uint offset = VT * tid + i + bLoadOffset;
			if(offset >= Capacity) offset -= Capacity;
			offset += offset / WARP_SIZE;
				
			b[i] = bData_shared[offset];
		}































////////////////////////////////////////////////////////////////////////////////
// LoadSegmentsCircular

// Read numSegments of data from data_global and store to the circular array
// data_shared. Elements in this array are also strided, so consecutive segments
// are spaced 33 slots apart. This allows for strided to thread order transpose.

template<typename T>
DEVICE2 void LoadSegmentsCircular(uint tid, uint numThreads, uint offset,
	uint capacity, uint count, const T* data_global) {

	for(int i = tid; i < count; i += numThreads) {
		T x = data_global[i];

		uint offset2 = offset + i;
		if(offset2 >= capacity)
			offset2 -= capacity;
		
		// Change to strided order by adding a misalignment slot every 32
		// elements.
		offset2 += offset2 / WARP_SIZE;

		data_shared[offset2] = x;
	}
}

// Search 256 threads with 4 values per thread.
template<int NumThreads, int VT, typename T>
DEVICE2 InsertResult<VT> Search256x4(uint tid, uint warp, uint lane,
	const T* aData_shared, const T* bData_shared, uint bOffset,
	uint bCapacity, T* keys_shared, const T b[VT], uint aCount, uint bCount, 
	uint* indices_shared) {

	const int NumValues = VT * NumThreads;

	// Take a multi-tiered approach to progressively narrow binary search
	// ranges:
	const int IndicesCapacity = NumThreads + NumThreads / WARP_SIZE;
	__shared__ uint indices_shared[IndicesCapacity];

	////////////////////////////////////////////////////////////////////////////
	// 

	if(tid < WARP_SIZE) {
		const uint Spacing = NumValues / WARP_SIZE;
		uint offset = bOffset +   
		T key = 


	}
	


}

template<int VT, typename T>
DEVICE2 InsertResult<VT> SearchInsertRange(uint tid, uint aCount, uint bCount,
	uint aRemaining, uint bRemaining, const T* aData_shared,
	const T* bData_shared, const T b[VT], int kind) {

}



template<int NumThreads, int VT, typename T>
DEVICE2 void SearchBlock(const T* aData_global, int2 aRange, 
	const T* bData_global, int2 bRange, int kind, int2* indices_global) {

	const int NumValues = NumThreads * VT;
	const int Spacing = WARP_SIZE + 1;
	const int Capacity = NumValues + WARP_SIZE;
	const int BufferSize = Capacity + Capacity / WARP_SIZE;

	__shared__ T aData_shared[NumValues]; 
	__shared__ T bData_shared[BufferSize];

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	uint warpValOffset = warp * WARP_SIZE * VT;

	int aLoaded = 0;

	// Round bData_global down to be aligned with the start of a segment.
	bData_global -= RoundDown(bRange.x, WARP_SIZE);

	// bStoreOffset is the offset in bData_shared (circular, strided) to write 
	// the next loaded segment.
	uint bStoreOffset = 0;

	// bLoadOffset is the offset in bData_shared corresponding to the next 
	// loaded value for tid 0.
	uint bLoadOffset = 0;

	// Number of individual elements available for reading starting at
	// bLoadOffset. Note that circular arithmetic is required to find the 
	// actual offsets.
	uint bLoaded = 0;

	// Prime the circular buffer by loading the first segment.
	if(bRange.x < bRange.y) {
		if(tid < WARP_SIZE)
			bData_shared[lane] = bData_global[lane];
		bData_global += WARP_SIZE;

		bStoreOffset = WARP_SIZE;
		bLoadOffset = (WARP_SIZE - 1) & bRange.x;
		bLoaded = WARP_SIZE - bLoadOffset;
	}

	uint aRemaining = aRange.y - aRange.x;
	uint bRemaining = bRange.y - bRange.x;

	InsertResult<VT> result;

	while(aRemaining || bRemaining) {
		
		////////////////////////////////////////////////////////////////////////
		// Load the values to fill the buffers.

		uint aCount = min(NumValues, aRemaining);
		uint bCount = min(NumValues, bRemaining);

		T a = (tid >= aLoaded) ? 
			aData_global[min(aRange.y - 1, aRange.x + tid)] : 
			aData_shared[tid + result.aConsume];

		// We need at least bCount loaded values in the b array yet we have only
		// bLoaded available. Compute the number of segments to pull in to give
		// us what is required.
		uint bLoadCount = RoundUp(bCount - bLoaded, WARP_SIZE);
		LoadSegmentsCircular(tid, NumThreads, bStoreOffset, Capacity,
			bLoadCount, bData_global);
		bData_global += bLoadCount;
		bStoreOffset += bLoadCount;
		if(bStoreOffset > Capacity) 
			bStoreOffset -= Capacity;
		bLoaded += bLoadCount;

		// Read VT elements from the circular array to register.
		T b[VT];

		// If this warp straddles the end of the circular buffer, we need to do
		// explicit range checking. This is expensive, so we try to branch out
		// of it for all the other warps.
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			uint offset = VT * tid + i + bLoadOffset;
			if(offset >= Capacity) offset -= Capacity;
			offset += offset / WARP_SIZE;
				
			b[i] = bData_shared[offset];
		}


		////////////////////////////////////////////////////////////////////////

		result = SearchInsertRange(tid, aCount, bCount, aRemaining - aCount,
			bRemaining - bCount, aData_shared, bData_shared, 















	}
}
		result = SearchInsertRange<T, T2>(tid, aCount, bCount, 
			aRemaining - aCount, bRemaining - bCount, a, b,
			aData_shared, bData_shared, radix_shared, NumRadixBits, kind);

		if(tid < result.bConsume)
			indices_global[bRange.x + tid] = 
				make_int2(result.index + aRange.x, b);


		////////////////////////////////////////////////////////////////////////
		// Advance the iterators by the consumed counts.

		// Update the offsets for the next serialization.
		aRange.x += result.aConsume;
		bRange.x += result.bConsume;
		
		aLoaded = aCount - result.aConsume;
		bLoaded = bCount - result.bConsume;

		aRemaining -= result.aConsume;
		bRemaining -= result.bConsume;
/*


template<int VT, typename T>
DEVICE2 void SearchInsertResults<VT> ThreadBinarySearch(uint tid, 
	const T* aData_shared, uint aCount, const T* bData, uint* indices) {

note: 


}



// VT = values per thread. Note that both A and B arrays have VT elements per
// thread. Make this large, but not so large that it hurts occupancy.
template<int NumThreads, int VT, typename T>
DEVICE2 void SearchBlock(const T* aData_global, int2 aRange,
	const T* bData_global, int2 bRange, int kind, int2* indices_global) {

	const int NumValues = NumThreads * VT;
	const int Space = NumValues + NumValues / WARP_SIZE;

	__shared__ T aData_shared[Space];
	
	uint tid = threadIdx.x;

	// aRange.x and bRange.y refer to the first element of each array in each
	// inner loop. They are not pointers to the next value to load. This
	// function streams at 

	// aLoaded and bLoaded are the counts of loaded but unprocessed values from
	// each stream.
	// aRemaining and bRemaining are the counts of unprocessed values from each
	// stream, including the loaded values.
	int aLoaded = 0;
	int bLoaded = 0;
	int aRemaining = aRange.y - aRange.x;
	int bRemaining = bRange.y - bRange.x;

	T a[VT], b[VT];

	SearchInsertResult result;

	while(aRemaining || bRemaining) {
		
		////////////////////////////////////////////////////////////////////////
		// Load the values to fill the buffers.

		int aCount = min(NumValues, aRemaining);
		int bCount = min(NumValues, bRemaining);

		// Move b values from register back into shared memory.
		#pragma unroll 
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i - result.bConsume;
			
			if(index >= 0) {


			}



		}






		T a = (tid >= aLoaded) ? 
			aData_global[min(aRange.y - 1, aRange.x + tid)] : 
			aData_shared[tid + result.aConsume];
		T b = (tid >= bLoaded) ?
			bData_global[min(bRange.y - 1, bRange.x + tid)] :
			bData_shared[tid + result.bConsume];
		__syncthreads();

		aData_shared[tid] = a;
		bData_shared[tid] = b;
		__syncthreads();

		result = SearchInsertRange<T, T2>(tid, aCount, bCount, 
			aRemaining - aCount, bRemaining - bCount, a, b,
			aData_shared, bData_shared, radix_shared, NumRadixBits, kind);

		if(tid < result.bConsume)
			indices_global[bRange.x + tid] = 
				make_int2(result.index + aRange.x, b);


		////////////////////////////////////////////////////////////////////////
		// Advance the iterators by the consumed counts.

		// Update the offsets for the next serialization.
		aRange.x += result.aConsume;
		bRange.x += result.bConsume;
		
		aLoaded = aCount - result.aConsume;
		bLoaded = bCount - result.bConsume;

		aRemaining -= result.aConsume;
		bRemaining -= result.bConsume;
	}
}*/