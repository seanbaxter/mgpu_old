#include "common.cu"



DEVICE int2 TestSortOrder(volatile uint* sorted_scratch, uint value, 
	uint bucket, uint bit, uint numBits, uint lane, uint pass) {

	// Alternate the scratch space so that the last thread's value of the 
	// preceding pass is always available. Store the full value so that we can
	// test the order of both the full key and of the radix digit.
	uint preceding;
	if(1 & pass) {
		sorted_scratch[WARP_SIZE] = value;
		preceding = sorted_scratch[WARP_SIZE - 1];
	} else {
		sorted_scratch[0] = value;
		preceding = sorted_scratch[lane ? -1 : 63];
	}
	uint precedingBucket = bfe(preceding, bit, numBits);

	return make_int2(preceding <= value, precedingBucket <= bucket);
}
