#include "../../../inc/mgpusearch.h"
#include "../../../util/cucpp.h"
#include <vector>
#include <algorithm>
#include <random>

typedef __int64 int64;

	
const int NumElements = 1000007;

std::tr1::mt19937 mt19937;

template<typename T>
void FillVec(std::vector<T>& vec, int count) {
	std::tr1::uniform_int<T> r(0, 50);
	T cur = 0;
	vec.resize(count);
	for(int i(0); i < count; ++i) {
		vec[i] = cur;
		if(0 == r(mt19937)) ++cur;
	}
}

int main(int argc, char** argv) {

	typedef int64 T;

	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	searchEngine_t engine = 0;
	searchStatus_t status = searchCreate("../../src/cubin/search.cubin",
		&engine);

	std::vector<T> values;
	FillVec(values, NumElements);

	DeviceMemPtr deviceData, deviceTree, deviceResults;
	context->MemAlloc(values, &deviceData);

	int treeSize = searchTreeSize(NumElements, SEARCH_TYPE_INT64);

	context->ByteAlloc(treeSize, &deviceTree);

	status = searchBuildTree(engine, NumElements, SEARCH_TYPE_INT64, 
		deviceData->Handle(), deviceTree->Handle());

	T last = values.back();

	// SEARCH
	const int NumQueries = 1000;
	T keys[NumQueries];
	for(int i(0); i < NumQueries; ++i) {
		float delta = (float)last / NumQueries;
		keys[i] = (int)(delta / 2 + i * delta);
	}
	keys[NumQueries - 1] = 1000000;

	DeviceMemPtr keysDevice, indicesDevice;
	context->MemAlloc(keys, NumQueries, &keysDevice);
	context->MemAlloc<uint>(NumQueries, &indicesDevice);
	status = searchKeys(engine, NumElements, SEARCH_TYPE_INT64, 
		deviceData->Handle(), SEARCH_ALGO_UPPER_BOUND, keysDevice->Handle(),
		NumQueries, deviceTree->Handle(), indicesDevice->Handle());

	std::vector<int> indicesHost;
	indicesDevice->ToHost(indicesHost);

	for(int i(0); i < NumQueries; ++i) {
		int j = indicesHost[i];
	//	printf("%I64d %d: (%I64d, %I64d)\n", keys[i], j, values[j - 1], values[j]);
		printf("%d\n", j);
	}

	searchDestroy(engine);
}

/*
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
}*/









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