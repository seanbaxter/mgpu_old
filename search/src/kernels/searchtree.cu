const int SearchTypeLower = 0;
const int SearchTypeUpper = 1;
const int SearchTypeRange = 2;

template<typename T>
struct BTree {
	const T* nodes[6];
	uint levelCounts[6];
	uint roundDown[6];
	uint numLevels;
	uint baseCount;
};


template<typename T> DEVICE2
uint GetOffset(T key, T node, int type, uint mask) {
	uint p;
	if(SearchTypeLower == type) p = key > node;
	if(SearchTypeUpper == type) p = key >= node;

	uint bits = mask & __ballot(p);
	uint offset = __popc(bits);

	return offset;
}

template<typename T> DEVICE2
uint DescendTree(uint offset, uint& o2, T& laneNode, T key, uint segLane,
	uint level, int type, uint mask, BTree<T> tree, int numLevels) {

	const int SegLanes = SEG_SIZE / sizeof(T);

	laneNode = tree.nodes[level][offset + segLane];

	if(level < numLevels - 1) {
		// We're in a b-tree level.
		o2 = GetOffset(key, laneNode, type, mask);
		offset = SegLanes * (offset + o2);
		offset = min(offset, tree.roundDown[level]);
	} else {
		// We're on the base level of data.
		if(offset + 32 > tree.baseCount) {
			// Change the mask if we're at the end of the data.
			uint setBits = tree.baseCount - offset;
			mask = bfi(0, 0xffffffff, 0, setBits);		
		}
		o2 = GetOffset(key, laneNode, type, mask);
		offset += o2;
	}
	return offset;
}

template<int Loop, typename T> __forceinline__ __device__
void SearchTree(const T* keys_global, uint numQueries, int type, BTree<T> tree,
	uint* indices_global, T* retrieved_global) {
	
	const int SegLanes = SEG_SIZE / sizeof(T);

	uint tid = threadIdx.x;
	uint block = blockIdx.x;
	uint segLane = (SegLanes - 1) & tid;
	uint warpLane = (WARP_SIZE - 1) & tid;
	uint node = tid / SegLanes;
	uint gid = block * (1024 / SegLanes) + node;
	uint stride = (1024 / SegLanes) * gridDim.x;
	
	uint mask = (4 == sizeof(T)) ? 0xffffffff :
		((warpLane >= SegLanes) ? 0xffff0000 : 0x0000ffff);
	
	// Load the root node and the first level of nodes.
	__shared__ T rootNode_shared[SegLanes];
	__shared__ T level1_shared[SegLanes * SegLanes];

	if(tid < SegLanes) rootNode_shared[tid] = tree.nodes[0][tid];
	if(tid < SegLanes * SegLanes) level1_shared[tid] = tree.nodes[1][tid];
	__syncthreads();

	T rootNode = rootNode_shared[segLane];

	for(int query = gid; query < numQueries; query += stride) {
		// TODO: preload the queries. Will eliminate a transaction per query.
		T key = keys_global[query];

		// Find the offset within the 16-way or 32-way tree node.
		uint offset = GetOffset(key, rootNode, type, mask);

		// Multiply by 16 or 32 to get the offset for the next node in the tree.
		// To prevent indexing past the end of the array, we round down to the
		// start of the last complete node in each level.
		offset = min(SegLanes * offset, tree.roundDown[0]);

		// Use the scaled offset + segLane to get this lane's comparison key.
		T laneNode = level1_shared[offset + segLane];
		uint o2 = 0;

		offset = DescendTree(offset, o2, laneNode, key, segLane, 1, type, mask,
			tree, Loop);

		// Advisory: Loop was not unrolled, unexpected call OPs.
		// Manually unroll the loop because nvcc is a POS.

		/*
		#pragma unroll
		for(int level = 2; level < Loop; ++level) {
			offset = DescendTree(offset, o2, laneNode, key, segLane, level, type,
				mask, tree);
		}
		*/


		if(2 < Loop) 
			offset = DescendTree(offset, o2, laneNode, key, segLane, 2, type,
				mask, tree, Loop);
		if(3 < Loop) 
			offset = DescendTree(offset, o2, laneNode, key, segLane, 3, type,
				mask, tree, Loop);
		if(4 < Loop) 
			offset = DescendTree(offset, o2, laneNode, key, segLane, 4, type,
				mask, tree, Loop);
		if(5 < Loop) 
			offset = DescendTree(offset, o2, laneNode, key, segLane, 5, type,
				mask, tree, Loop);

		if(!segLane) {

			indices_global[query] = offset;

		}
	}
}


extern "C" __global__ __launch_bounds__(1024, 1)
void SearchTreeIntLower(const int* keys, uint numQueries, BTree<int> tree,
	uint* indices_global, int* retrieved_global) {

	if(2 == tree.numLevels)
		SearchTree<2>(keys, numQueries, SearchTypeLower, tree, indices_global,
			retrieved_global);
	if(3 == tree.numLevels)
		SearchTree<3>(keys, numQueries, SearchTypeLower, tree, indices_global,
			retrieved_global);
	if(4 == tree.numLevels)
		SearchTree<4>(keys, numQueries, SearchTypeLower, tree, indices_global,
			retrieved_global);
	if(5 == tree.numLevels)
		SearchTree<5>(keys, numQueries, SearchTypeLower, tree, indices_global,
			retrieved_global);
}
