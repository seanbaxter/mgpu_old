#include "ranges.cu"


////////////////////////////////////////////////////////////////////////////////
// SearchBLockConstricted

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

	// PROCESS 32 THREADS.

	// Run elements Spacing * tid. For NumValues = 1024, these are elements:
	// 0, 32, 64, 96, 128, etc.
	if(tid < WARP_SIZE) {
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
	}
	__syncthreads();

	// PROCESS 32 THREADS (64 done).

	// Run elements Spacing * tid + Spacing / 2. We use the indices to the left
	// and right (i.e. Spacing * tid and Spacing * tid + Spacing) to constrain
	// this search. For NumValues = 1024, these are elements:
	// 16, 48, 80, 112, 144, etc.
	if(tid < WARP_SIZE) {
		int j = Spacing2 * tid;
		j += j / WARP_SIZE;
		int j2 = Spacing2 * tid + Spacing2;
		j2 += j2 / WARP_SIZE;

		uint begin = indices_shared[j];
		uint end = indices_shared[j2];

		int i = Spacing * tid + Spacing / 2;
		i += i / WARP_SIZE;

		T key = bData_shared[i2];
		uint index = RangeBinarySearch(aData_shared, begin, end, key, kind, 
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
DEVICE2 void SearchThread4(uint tid, const T* aData_shared, 
	const uint* indices_shared, uint bCount, const uint* indices_shared,
	const T keys[4], uint indices[4], uint* target_shared, int store) {

	uint activeThreads = DivUp(bCount, 4);
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
			target_shared[4 * tid + 1 + index1] = 4 * tid + 1;
			target_shared[4 * tid + 1 + index2] = 4 * tid + 2;
			target_shared[4 * tid + 1 + index3] = 4 * tid + 3;
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

	int last = bCount - 1;
	last += last / WARP_SIZE;
	int index = min(33 * tid + 31, last); 
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


template<typename T>
DEVICE2 uint FindAIntoB(uint tid, const T* aData_shared, uint aCount, T bLast,
	int kind) {


}


/////////////////////////////////////////////////////////////////////////////////



















		// Load the last value of both arrays to establish the consumed
		// counts.
		params.a = aData;
		params.b = bData;
		params.aLast = aData_shared[aCount - 1];
		params.bLast = bData_shared[bCount - 1];
		params.aPrev = aData_shared[tid - 1];
		params.bPrev = bData_shared[tid - 1];

		// Because we're only dealing with fragments of the datasets, we
		// need to be careful on which values can be inserted. Take the
		// upper_bound case as an example:
		
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 2 3 4 7 8 8 9 ...
		// bData = 1 2 2 2 6 6 9 9 ...

		// We can insert elements with values 1 - 6 from bData without 
		// ambiguity. However we can't be sure where to insert the pair of
		// 9s. The upper_bound semantics demand that they be inserted AFTER
		// all occurences of the same value in the aData stream. Because we
		// haven't seen the next elements in aData we don't know if we can
		// insert bData[6] and bData[7] at i = 8.

		// The same problem occurs with lower_bound: 
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 2 3 4 7 7 7 8 ... 
		// bData = 1 2 2 2 6 9 9 9 ...

		// aData[8] could be an 8, in which case it would precede bData[5],
		// or it could be a 9 or something larger, in which case the 9 terms
		// from bData should come first.

		// The problem is symmetric even when we only want to find insert 
		// points from bData into aData:
		//     i = 0 1 2 3 4 5 6 7
		// aData = 2 3 3 3 5 6 7 7 ...
		// bData = 1 1 2 3 3 3 3 4 (4 5 5 6 7 7 7)

		// The goal is to consume both the aData and bData arrays in shared
		// memory every iteration. However we need to retain aData values
		// when they are required to know where to insert upcoming bData
		// values. In the example above, we can consume all 8 values in 
		// bData, but only the first 4 values in aData. The remaining aData
		// values are shifted forward in aData_shared and are used for
		// supporting the search with the subsequent bData values.

		// When searching with lower_bound, if both aData and bData shared
		// memory arrays end with the same value, both arrays are completely
		// consumed.

		// At least one of the arrays will be completely consumed. To
		// summarize, when inserting bData into aData, lower_bound favors
		// consuming bData and upper_bound favors consuming aData.

		if(aRemaining + bRemaining) {
			// Compare the last element in bData to the last element in aData.
			bool pred = kind ? (params.bLast < params.aLast) : 
				(params.bLast <= params.aLast);

			// Get the preceding keys to find the first value that violates the
			// consume conditions.
			if(pred) {
				// If the last element in bData is < (or <=) the last element in
				// aData, completely consume the bData_shared stream.

				// If upper_bound consume all values in aData that are <= bLast.
				// If lower_bound consume all values in aData that are < bLast.
				bool inRange = kind ? (params.a <= params.bLast) :
					(params.a < params.bLast);
				bool inRangePrev = kind ? (params.aPrev <= params.bLast) :
					(params.aPrev < params.bLast);
				if(!tid) inRangePrev = true;
				if(!inRange && inRangePrev)
					consumed_shared = tid;

				__syncthreads();
				result.aConsume = consumed_shared;
			} else {
				// If the last element in bData is not < (or <=) the last
				// element in aData, completely consume the aData_shared stream.

				// If upper_bound consume all values in bData that are <= aLast.
				// If lower_bound consume all values in bData that are < aLast.
				bool inRange = kind ? (params.b < params.aLast) :
					(params.b <= params.aLast);
				bool inRangePrev = kind ? (params.bPrev < params.aLast) :
					(params.bPrev <= params.aLast); 
				if(!tid) inRangePrev = true;
				if(!inRange && inRangePrev)
					consumed_shared = tid;

				__syncthreads();
				result.bConsume = consumed_shared;
			}
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