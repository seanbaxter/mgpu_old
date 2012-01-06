#include "ranges.cu"

template<int VT>
struct InsertResult {
	uint indices[VT];
	uint aConsume, bConsume;
};


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