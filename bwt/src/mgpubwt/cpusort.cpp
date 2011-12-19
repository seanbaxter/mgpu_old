#include "bwtsort.h"
#include <cassert>
#include <algorithm>

// 
// http://en.wikipedia.org/wiki/Quicksort#In-place_version

struct PartitionResult {
	int* index;
	int leftMatch;
	int rightMatch;
};

inline void FastSwap(int* a, int* b) {
	int x = *a;
	*a = *b;
	*b = x;
}

inline PartitionResult Partition(int* left, int* pivot, int* right, 
	const byte* symbols, int count2) {

	PartitionResult result;
	result.leftMatch = 0;//count2;
	result.rightMatch = 0;//count2;
	result.index = left;
	
	const byte* b = symbols + *pivot;
	FastSwap(pivot, right);

	for(int* i = left; i < right; ++i) {
		// Compare the string at symbols + *i with the string at
		// symbols + *pivot. We compare at most count2 items. Keep track of how
		// many characters match.
		const byte* a = symbols + *i;

		int cmp = memcmp(a, b, count2);
		if(cmp < 0) {
			FastSwap(i, result.index);
			++result.index;
		}
/*
		for(int j(0); j < count2; ++j) {
			byte a2 = a[j];
			byte b2 = b[j];

			if(a2 != b2) {
				if(a2 < b2) {
					result.leftMatch = std::min(result.leftMatch, j);
					FastSwap(i, result.index);
					++result.index;
				} else 
					result.rightMatch = std::min(result.rightMatch, j);
				break;
			}
		}*/
	}
	FastSwap(result.index, right);
	return result;
}

void QSortStrings(int* left, int* right, const byte* symbols, int count2) {
	assert(left < right);

	// If there are only two elements do the memcmp and return.
	if(right - left == 1) {
		int cmp = memcmp(symbols + *left, symbols + *right, count2);
		if(cmp > 0) FastSwap(left, right);
		return;
	}

	// Choose the midpoint for the pivot.
	int* pivot = left + (right - left) / 2;

	PartitionResult partition = Partition(left, pivot, right, symbols, count2);

	// Examine the left partition and recurse if there are multiple elements.
	int* r2 = partition.index - 1;
	if(left < r2)
		QSortStrings(left, r2, symbols + partition.leftMatch, 
			count2 - partition.leftMatch);
	
	// Examine the right partition and recurse if there are multiple elements.
	int* l2 = partition.index + 1;
	if(l2 < right)
		QSortStrings(l2, right, symbols + partition.rightMatch,
			count2 - partition.rightMatch);
}


/*
struct Pred {
	const byte* symbols;
	int count2;
	bool operator()(int a, int b) const {
		int cmp = memcmp(symbols + a, symbols + b, count2);
		return cmp < 0;
	}
};

void QSortStrings(int* left, int* right, const byte* symbols, int count2) {
	Pred pred;
	pred.symbols = symbols;
	pred.count2 = count2;
	std::sort(left, right + 1, pred);
}*/