#include <cstdio>
#include <vector>

int main() {
	int WarpSize = 32;
	int ValuesPerThread = 4;

	const int NumValues = WarpSize * ValuesPerThread;

	std::vector<int> v1(33 * WarpSize * ValuesPerThread), v2(NumValues);
	std::vector<int> b1(NumValues), b2(NumValues);

	for(int i(0); i < NumValues; ++i) {
		// Store into v1.
		int lane = i % WarpSize;
		int warp = i / WarpSize;

		v1[lane + warp * (WarpSize + 1)] = i;
		b1[i] = (lane + warp * (WarpSize + 1)) % 32;
	}

	for(int i(0); i < NumValues; ++i) {
		int lane = i % WarpSize;
		int warp = i / WarpSize;
		
		int offset = ValuesPerThread * lane;
		offset += offset / WarpSize;

		v2[i] = v1[offset + warp];
		b2[i] = (offset + warp) % 32;

	}


	int i = 0;

}