#include <random>
#include <vector>

typedef unsigned int uint;

std::tr1::mt19937 mt19937;


////////////////////////////////////////////////////////////////////////////////
// Sort1 is a simple CPU radix sort. It is not a parallel implementation, as the
// calculation of relative offsets is sequential.

void Sort1Pass(uint* keys, uint bit, uint numBits, size_t count) {
	uint numBuckets = 1<< numBits;
	uint mask = numBuckets - 1;

	// Compute the frequency of occurence for each key. This corresponds to the
	// count kernel in the GPU version.
	std::vector<uint> counts(numBuckets);
	for(size_t i(0); i < count; ++i) {
		uint key = mask & (keys[i]>> bit);
		++counts[key];			// Simply increment.
	}

	// Iterate over each bucket and compute the exclusive scan. This corresponds
	// to the hist1 and hist2 kernels in the GPU evrsion.
	std::vector<uint> excScan(numBuckets);
	uint prev = 0;
	for(uint i(0); i < numBuckets; ++i) {
		// Assign the total number of keys encountered at the previous bucket.
		excScan[i] = prev;

		// Increment the counter to include the counts in this bucket.
		prev += counts[i];
		
		// Clear the counters for the next pass.
		counts[i] = 0;
	}

	// Iterate over the keys again, and add their relative offsets to the
	// excScan offset corresponding to their bucket. In the GPU version, 
	// this work is accomplished with the histogram and sort passes.
	std::vector<uint> sorted(count);
	for(size_t i(0); i < count; ++i) {
		uint key = mask & (keys[i]>> bit);

		// Retrieve and running count of keys in this bucket.
		uint rel = counts[key];

		// Add the exclusive scan of total key counts with the relative offset
		// to get the scatter offset.
		uint scatter = excScan[key] + rel;
		sorted[scatter] = keys[i];

		// Increment the running count.
		counts[key] = rel + 1;
	}

	// Copy the sorted array into keys and return.
	std::copy(sorted.begin(), sorted.end(), keys);
}

void Sort1(uint* keys, size_t count) {
	// Make 8 sort passes of 4 bits each.
	for(int bit(0); bit < 32; bit += 4)
		Sort1Pass(keys, bit, 4, count);
}


////////////////////////////////////////////////////////////////////////////////
// Sort2 is a CPU radix sort that demonstrates some parallelism. The key array
// is broken into blocks, which are processed independently (they can be farmed
// to different threads, or in the GPU domain, different thread blocks). A 
// reduction pass takes the block counts and sums them for global counts and
// relative offsets. The blocks are then processed independently once again,
// and scatter like the sequential version.

// Returns the count for each radix digit in the block. Corresponds to the count
// kernel in the CUDA implementation.
void Sort2CountPass(const uint* keys, uint bit, uint numBits, size_t count, 
	uint* counts) {

	// Compute the histogram of radix digits for this block only. This
	// corresponds exactly to the count kernel in the GPU implementation.
	uint mask = (1<< numBits) - 1;
	for(size_t i(0); i < count; ++i) {
		uint key = mask & (keys[i]>> bit);
		++counts[key];
	}
}

// Adds the relative offsets for each digit within the block with the global
// offset of each digit and scatters. Corresponds to the sort kernel in the
// CUDA implementation.
void Sort2SortPass(const uint* keys, uint* sorted, uint bit, uint numBits,
	size_t count, const uint* digitOffsets) {

	// Compute the histogram of radix digits for this block only. This
	// corresponds exactly to the count kernel in the GPU implementation.
	uint numBuckets = 1<< numBits;
	uint mask = numBuckets - 1;
	std::vector<uint> counts(numBuckets);

	for(size_t i(0); i < count; ++i) {
		// Extract the key 
		uint key = mask & (keys[i]>> bit);
		uint rel = counts[key];
		uint scatter = rel + digitOffsets[key];

		sorted[scatter] = keys[i];

		counts[key] = 1 + rel;
	}
}

// Runs a three-phase histogram sort.
// 1) For each block, compute that block's digit count into its own section of
// countsBlock. This part can be done in parallel.
// 2a) 
void Sort2Pass(uint* keys, uint bit, uint numBits, size_t count, 
	uint blockSize) {

	uint numBuckets = 1<< numBits;
	uint numBlocks = (count + blockSize - 1) / blockSize;
	std::vector<uint> countsBlock(numBuckets * numBlocks);

	// Compute the radix digit histogram for each block. This operation can be
	// executed in parallel.
	for(uint block(0); block < numBlocks; ++block) {
		uint index = block * blockSize;
		Sort2CountPass(keys + index, bit, numBits, 
			std::min(count - index, blockSize),
			&countsBlock[block * numBuckets]);
	}

	// This reduction pass sums counts together and begins computing the key
	// relative offsets. This corresponds to hist1 and hist2.
	std::vector<uint> countsGlobal(numBuckets),
		excScanBlock(numBlocks * numBuckets);
	for(uint block(0); block < numBlocks; ++block) {
		for(uint i(0); i < numBuckets; ++i) {
			uint index = block * numBuckets + i;
			excScanBlock[index] = countsGlobal[i];
			countsGlobal[i] += countsBlock[index];
		}
	}

	// Compute the exclusive scan of the count totals. This corresponds to 
	// hist2.
	std::vector<uint> excScanGlobal(numBuckets);
	uint prev = 0;
	for(uint i(0); i < numBuckets; ++i) {
		excScanGlobal[i] = prev;
		prev += countsGlobal[i];
	}

	// Sort each block in parallel. Add excScanGlobal to excScanBlock for each
	// bucket, then scatter the values to the target array.
	std::vector<uint> sorted(count);
	for(uint block(0); block < numBlocks; ++block) {
		std::vector<uint> bucketOffsets(numBuckets);
		for(uint i(0); i < numBuckets; ++i)
			bucketOffsets[i] = excScanGlobal[i] +
				excScanBlock[block * numBuckets + i];
		
		uint index = block * blockSize;
		Sort2SortPass(keys + index, &sorted[0], bit, numBits, 
			std::min(count - index, blockSize), &bucketOffsets[0]);
	}

	// Copy the sorted array into keys and return.
	std::copy(sorted.begin(), sorted.end(), keys);
}

void Sort2(uint* keys, size_t count, uint blockSize) {
	// Make 8 sort passes of 4 bits each.
	for(int bit(0); bit < 32; bit += 4)
		Sort2Pass(keys, bit, 4, count, blockSize);
}


int main(int argc, char** argv) {
	const size_t NumElements = 1024;
	std::vector<uint> keys(NumElements);

	std::tr1::uniform_int<uint> r(0, 0xffffffff);
	for(size_t i(0); i < NumElements; ++i)
		keys[i] = r(mt19937);

	Sort2(&keys[0], NumElements, 64);
	for(int i(0); i < NumElements; ++i) {
		printf("0x%08x ", keys[i]);
		if((3 & i) == 3) printf("\n");
	}
}

