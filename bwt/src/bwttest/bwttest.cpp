#include "../../../inc/mgpubwt.h"
#include "../../../util/cucpp.h"
#include <cstdio>
#include <fstream>
#include <string>

const char* CubinPath = "../../src/cubin/";

const char* TestFile = "../../src/mobydick.txt";

int main(int argc, char** argv) {
	cuInit(0);

	DevicePtr device;
	CUresult result = CreateCuDevice(0, &device);
	if(CUDA_SUCCESS != result || 2 != device->ComputeCapability().first) {
		printf("Could not create CUDA 2.0 device.\n");
		return 0;
	}

	ContextPtr context;
	result = CreateCuContext(device, 0, &context);
	
	bwtEngine_t engine;
	bwtStatus_t status = bwtCreateEngine(CubinPath, &engine);

	std::ifstream f(TestFile);
	if(!f.good()) {
		printf("Could not open file %s.\n", TestFile);
		return 0;
	}

	std::vector<char> data;
	while(f.good()) {
		std::string line;
		std::getline(f, line);

		if(line.size()) {
			data.insert(data.end(), line.begin(), line.end());
			data.push_back(' ');
		} else
			data.push_back('\n');		
	}
//	std::string s((std::istreambuf_iterator<char>(f)),
//		std::istreambuf_iterator<char>());

	int count = data.size();
	// fwrite(&data[0], 1, 100000, stdout);

	std::vector<int> indices(count);
	std::vector<char> symbols(count);
	for(int keySize(16); keySize <= 24; ++keySize) {

		int numSegs;
		float avSegLength;
		status = bwtSortBlock(engine, &data[0], count, keySize, &symbols[0],
			&indices[0], &numSegs, &avSegLength);
		printf("key size = %2d   numSegs = %7d   avSegLen = %10.3f   ", keySize,
			numSegs, avSegLength);


		printf("Verifying...\n");
		std::vector<char> symbols2(2 * count);
		memcpy(&symbols2[0], &data[0], count);
		memcpy(&symbols2[0] + count, &data[0], count);

		for(int i(1); i < count; ++i) {
			// Test that the substrings are well-ordered.
			int indexPrev = indices[i - 1];
			int index = indices[i];

			const char* a = &symbols2[0] + indexPrev;
			const char* b = &symbols2[0] + index;
			int result = memcmp(a, b, count);

			if(result > 0) {
				printf("Error on i = %d.\n", i);
				return 0;
			}
		}
	}


return 0;



//	return 0;

	/*
	for(int i(0); i < count; ++i) {
		int index = indices[i];

		// print ten chars in front of and 70 characters after.
		for(int j(-10); j < 70; ++j) {
			int i2 = index + j;
			char c = ' ';
			if(i2 >= 0 && i2 < count && !isspace(data[i2]) && isprint(data[i2]))
				c = data[i2];
			printf("%c", c);
		}
		printf("\n");
	}*/

	for(int i(0); i < count; ++i) {
		char c = symbols[i];
		if(c != ' ' && isspace(c) || !isprint(c)) c = '*';
		printf("%c", c);
		if(79 == (i % 80)) printf("\n");
	}



	bwtDestroyEngine(engine);

}
