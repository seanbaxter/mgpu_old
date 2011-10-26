#include <cuda.h>
#include <cstdio>
#include <vector>
#include <cassert>

const int WarpSize = 32;

// If not power of 2, 

int IsConflict(int valuesPerThread, int stride) {
	const int NumValues = WarpSize * valuesPerThread;

	int Padding = stride - WarpSize;
	std::vector<int> v1(stride * valuesPerThread), v2(NumValues);
	std::vector<int> b1(NumValues), b2(NumValues);


	for(int i(0); i < NumValues; ++i) {
		// Store into v1.
		int lane = i % WarpSize;
		int warp = i / WarpSize;

		int offset = warp * WarpSize + lane;
		offset += Padding * (offset / WarpSize);

		v1[offset] = i;
		b1[i] = offset % 32;
	}

	for(int i(0); i < NumValues; ++i) {
		int lane = i % WarpSize;
		int warp = i / WarpSize;
		
		int offset = valuesPerThread * lane + warp;
		offset += Padding * (offset / WarpSize);

		int value = v1[offset];
		assert(value == lane * valuesPerThread + warp);

		v2[i] = v1[offset];
		b2[i] = (offset) % 32;
	}

	int maxConflict = 0;
	for(int warp(0); warp < valuesPerThread; ++warp) {
		std::vector<int> conflicts(32);
		int warpConflict = 0;
		for(int i(0); i < 32; ++i) {
			int bank = b2[32 * warp + i];
			++conflicts[bank];
			warpConflict = std::max(warpConflict, conflicts[bank]);
		}
		maxConflict = std::max(maxConflict, warpConflict);
	}
	return maxConflict;
}



int main() {

	for(int vt(1); vt <= 20; ++vt) {
		int bestStride = 0;
		int bestConflict = 100;
		for(int stride(32); stride < 40; ++stride) {
			int conflict = IsConflict(vt, stride);
			if(conflict < bestConflict) {
				bestStride = stride;
				bestConflict = conflict;
			}
		}
		printf("%d: %d (%d)\n", vt, bestStride, bestConflict);		
	}

	int i = 0;

}
