#include "../../../util/cucpp.h"
#include "../../../inc/mgpusearch.h"
#include <vector>
#include <algorithm>
#include <random>

template<typename T>
struct BTree {
	CUdeviceptr nodes[6];
	int levelCounts[6];
	uint roundDown[6];
	uint numLevels;
	uint baseCount;
};


////////////////////////////////////////////////////////////////////////////////
// Support functions for building and navigating GPU b-trees.

// Returns the size of each type.
const int TypeSizes[6] = { 4, 4, 4, 8, 8, 8 };


// Returns the number of active levels and their sizes.
int DeriveLevelSizes(int count, searchType_t type, int* sizes) {
	int size = TypeSizes[type];
	int SegLanes = 128 / size;
	
	int level = 1;
	sizes[0] = RoundUp(count, SegLanes);
	do {
		count = DivUp(count, SegLanes);
		count = RoundUp(count, SegLanes);
		sizes[level++] = count;
	} while(count > SegLanes);

	for(int i(0); i < level / 2; ++i)
		std::swap(sizes[i], sizes[level - 1 - i]);

	return level;
}

struct searchEngine_d {
	ContextPtr context;
	ModulePtr module;
	FunctionPtr search[6][4];
	FunctionPtr build[2];		// 32- and 64-bit versions.
};


////////////////////////////////////////////////////////////////////////////////
// Create and destroy the sparseEngine_d object.

searchStatus_t SEARCHAPI searchCreate(const char* kernelPath,
	searchEngine_t* engine) {

	std::auto_ptr<searchEngine_d> e(new searchEngine_d);

	// Get the current context and test the device version.
	CUresult result = AttachCuContext(&e->context);
	if(CUDA_SUCCESS != result) return SEARCH_STATUS_INVALID_CONTEXT;
	if(2 != e->context->Device()->ComputeCapability().first)
		return SCAN_STATUS_UNSUPPORTED_DEVICE;

	// Load the module.
	result = e->context->LoadModuleFilename(kernelPath, &e->module);
	if(CUDA_SUCCESS != result) return SEARCH_STATUS_KERNEL_NOT_FOUND;

	// Load the functions.

	result = e->module->GetFunction("BuildTree4", make_int3(256, 1, 1), 
		&e->build[0]);
	result = e->module->GetFunction("BuildTree8", make_int3(256, 1, 1), 
		&e->build[1]);

	result = e->module->GetFunction("SearchTreeIntLower", make_int3(1024, 1, 1),
		&e->search[0][0]);

	*engine = e.release();
	return SEARCH_STATUS_SUCCESS;
}

searchStatus_t SEARCHAPI searchDestroy(searchEngine_t engine) {
	delete engine;
	return SEARCH_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Build the search b-tree.

int SEARCHAPI searchTreeSize(int count, searchType_t type) {
	int sizes[6];
	int levels = DeriveLevelSizes(count, type, sizes);
	int total = 0;
	for(int i(0); i < levels - 1; ++i)
		total += sizes[i];
	
	return TypeSizes[type] * total;
}


searchStatus_t SEARCHAPI searchBuildTree(searchEngine_t engine, int count, 
	searchType_t type, CUdeviceptr data, CUdeviceptr tree) {

	// Build the tree from the bottom up.
	int sizes[6];
	int levels = DeriveLevelSizes(count, type, sizes);
	int size = TypeSizes[type];

	CUdeviceptr levelStarts[6];
	for(int i(0); i < levels - 1; ++i) {
		levelStarts[i] = tree;
		tree += size * sizes[i];
	}
	levelStarts[levels - 1] = data;
	sizes[levels - 1] = count;

	for(int i(levels - 2); i >= 0; --i) {
		CuCallStack callStack;
		callStack.Push(levelStarts[i + 1], sizes[i + 1], levelStarts[i]);

		int numBlocks = DivUp(sizes[i], 256);
		CUresult result = engine->build[size / 4 - 1]->Launch(numBlocks, 1,
			callStack);
	}

	return SEARCH_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Search the b-tree.

searchStatus_t SEARCHAPI searchKeys(searchEngine_t engine, int count,
	searchType_t type, CUdeviceptr data, searchAlgo_t algo, CUdeviceptr keys, 
	int numQueries, CUdeviceptr tree, CUdeviceptr results) {

	BTree<int> btree;
	btree.numLevels = DeriveLevelSizes(count, type, btree.levelCounts);
	int offset = 0;
	int size = TypeSizes[type];
	int segLanes = 128 / size;

	for(uint i(0); i < btree.numLevels - 1; ++i) {
		btree.nodes[i] = tree;
		btree.roundDown[i] = btree.levelCounts[i + 1] - segLanes;
		tree += btree.levelCounts[i] * size;
	}
	btree.baseCount = count;
	btree.nodes[btree.numLevels - 1] = data;
	btree.levelCounts[btree.numLevels - 1] = count;

	CuCallStack callStack;
	callStack.Push(keys, numQueries);
	callStack.PushStruct(btree, sizeof(CUdeviceptr));
	callStack.Push(results, (CUdeviceptr)0);

	const int segsPerBlock = 1024 / (128 / size);
	int numBlocks = DivUp(numQueries, segsPerBlock);
	numBlocks = std::min(numBlocks, engine->context->Device()->NumSMs());
	CUresult result = engine->search[0][0]->Launch(numBlocks, 1, callStack);
	
	return SEARCH_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////


struct BTreeCPU {
	int count;
	std::vector<int> data;	

	// Support up to 6 btree levels.
	int numLevels;
	int levelCounts[6];
	std::vector<int> levelData[6];
};

void CreateBTreeCPU(std::vector<int>& data, int count,
	std::auto_ptr<BTreeCPU>* ppTree) {

	std::auto_ptr<BTreeCPU> tree(new BTreeCPU);
	tree->numLevels = 0;
	tree->count = count;
	tree->data.swap(data);

	const int SEG_SIZE = 32;

	int level = 0;
	while(count > SEG_SIZE) {
		// Divide by 32 to get the size of the next btree level.
		int count2 = (count + SEG_SIZE - 1) / SEG_SIZE;

		// Round up to a multiple of 32 to make indexing simpler.
		int newCount = ~(SEG_SIZE - 1) & (count2 + SEG_SIZE - 1);
		tree->levelData[level].resize(newCount);

		// Prepare the subsampling.
		const int* source = level ? &tree->levelData[level - 1][0] :
			&tree->data[0];

		for(int i(0); i < newCount; ++i) {
			int j = std::min(SEG_SIZE * i + SEG_SIZE - 1, count - 1);
			tree->levelData[level][i] = source[j];
		}

		// Store the level count.
		tree->levelCounts[level++] = newCount;
		count = newCount;
	}
	tree->numLevels = level;

	// Swap the levels to put them in order.
	for(int i(0); i < level / 2; ++i) {
		tree->levelData[i].swap(tree->levelData[level - 1 - i]);
		std::swap(tree->levelCounts[i], tree->levelCounts[level - 1 - i]);
	}

	*ppTree = tree;
}


int main(int argc, char** argv) {

	
	std::tr1::mt19937 mt19937;

	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	searchEngine_t engine = 0;
	searchStatus_t status = searchCreate("../../src/cubin/search.cubin",
		&engine);

	const int NumElements = 20;
	std::vector<int> values(NumElements);

	std::tr1::uniform_int<int> r(0, 99);
	for(int i(0); i < NumElements; ++i)
		values[i] = r(mt19937);
	std::sort(values.begin(), values.end());

	DeviceMemPtr deviceData, deviceTree, deviceResults;
	context->MemAlloc(values, &deviceData);

	int treeSize = searchTreeSize(NumElements, SEARCH_TYPE_INT32);

	context->ByteAlloc(treeSize, &deviceTree);

	status = searchBuildTree(engine, NumElements, SEARCH_TYPE_INT32, 
		deviceData->Handle(), deviceTree->Handle());

	std::vector<int> hostTree;
	deviceTree->ToHost(hostTree);

	std::auto_ptr<BTreeCPU> btreeCPU;
	CreateBTreeCPU(values, NumElements, &btreeCPU);


	// SEARCH
	const int NumQueries = 10;
	int SearchKeys[NumQueries] = { 5, 15, 25, 35, 45, 55, 65, 75, 85, 95 };
	DeviceMemPtr keysDevice, indicesDevice;
	context->MemAlloc(SearchKeys, 10, &keysDevice);
	context->MemAlloc<uint>(10, &indicesDevice);
	status = searchKeys(engine, NumElements, SEARCH_TYPE_INT32, 
		deviceData->Handle(), SEARCH_ALGO_LOWER_BOUND, keysDevice->Handle(),
		NumQueries, deviceTree->Handle(), indicesDevice->Handle());

	std::vector<int> indicesHost;
	indicesDevice->ToHost(indicesHost);

	searchDestroy(engine);
}


/*
searchStatus_t SEARCHAPI searchKeys(searchEngine_t engine, int count,
	searchType_t type, CUdeviceptr data, searchAlgo_t algo, CUdeviceptr keys, 
	int numQueries, CUdeviceptr tree, CUdeviceptr results) {
*/
/*
#include <cstdio>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>

std::tr1::mt19937 mt19937;

// Build 

const int SEG_SIZE = 32;

struct BTree {
	int count;
	std::vector<int> data;	

	// Support up to 6 btree levels.
	int numLevels;
	int levelCounts[6];
	std::vector<int> levelData[6];
};

void CreateBTree(std::vector<int>& data, int count,
	std::auto_ptr<BTree>* ppTree) {

	std::auto_ptr<BTree> tree(new BTree);
	tree->numLevels = 0;
	tree->count = count;
	tree->data.swap(data);

	int level = 0;
	while(count > SEG_SIZE) {
		// Divide by 32 to get the size of the next btree level.
		int count2 = (count + SEG_SIZE - 1) / SEG_SIZE;

		// Round up to a multiple of 32 to make indexing simpler.
		int newCount = ~(SEG_SIZE - 1) & (count2 + SEG_SIZE - 1);
		tree->levelData[level].resize(newCount);

		// Prepare the subsampling.
		const int* source = level ? &tree->levelData[level - 1][0] :
			&tree->data[0];

		for(int i(0); i < newCount; ++i) {
			int j = std::min(SEG_SIZE * i + SEG_SIZE - 1, count - 1);
			tree->levelData[level][i] = source[j];
		}

		// Store the level count.
		tree->levelCounts[level++] = newCount;
		count = newCount;
	}
	tree->numLevels = level;

	// Swap the levels to put them in order.
	for(int i(0); i < level / 2; ++i) {
		tree->levelData[i].swap(tree->levelData[level - 1 - i]);
		std::swap(tree->levelCounts[i], tree->levelCounts[level - 1 - i]);
	}

	*ppTree = tree;
}

int GetOffset(int key, const int* node) {
	for(int i(0); i < SEG_SIZE; ++i)
		if(node[i] >= key) return i;
	return SEG_SIZE;
}
int GetOffset2(int key, const int* node, int offset, int count) {
	int end = std::min(offset + SEG_SIZE, count);
	for(int i(offset); i < end; ++i)
		if(node[i] >= key) return i;
	return end;
}

int lower_bound(const BTree& tree, int key) {
	int numLevels = tree.numLevels;
	int offset = 0;
	for(int level(0); level < numLevels; ++level) {
		int o2 = GetOffset(key, &tree.levelData[level][offset]);
		offset = SEG_SIZE * (offset + o2);
	}
	offset = GetOffset2(key, &tree.data[0], offset, tree.count);
	return offset;	
}

int main(int argc, char** argv) {
	const int NumElements = 20000;
	std::tr1::uniform_int<int> r(0, 32767);

	std::vector<int> data(NumElements);
	for(int i(0); i < NumElements; ++i)
		data[i] = r(mt19937);
	std::sort(data.begin(), data.end());

	std::auto_ptr<BTree> tree;
	CreateBTree(data, NumElements, &tree);

	int offset = lower_bound(*tree, 32700);

	return 0;
}
*/