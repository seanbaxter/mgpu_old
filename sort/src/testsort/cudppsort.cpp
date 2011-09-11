#include "benchmark.h"
#include "../cudpp2/cudpp_manager.h"
#include "../cudpp2/cudpp_radixsort.h"


// Use our specially hacked CUDPP and benchmark with B40C's terms.
CUresult CUDPPBenchmark(CUDPPHandle handle, B40cTerms& terms,
	double* elapsed) {

	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.op = CUDPP_OPERATOR_INVALID;
	config.datatype = CUDPP_UINT;
	config.options = terms.randomVals ? CUDPP_OPTION_KEY_VALUE_PAIRS :
		CUDPP_OPTION_KEYS_ONLY;

	CUDPPManager* manager = CUDPPManager::getManagerFromHandle(handle);
	std::auto_ptr<CUDPPRadixSortPlan> plan(new CUDPPRadixSortPlan(manager,
		config, terms.count));
	plan->m_keyBits = terms.numBits;

	CuEventTimer timer;
	for(int i(0); i < terms.iterations; ++i) {
		if(!i || terms.reset) {
			timer.Stop();

			terms.randomKeys->ToDevice(terms.sortedKeys);
			if(terms.randomVals)
				terms.randomVals->ToDevice(terms.sortedVals);
		}
		timer.Start(false);

		void* keys, *vals;
		if(terms.reset) {
			keys = (void*)terms.sortedKeys->Handle();
			vals = terms.sortedVals ? (void*)terms.sortedVals->Handle() : 0;			
		} else {
			keys = (void*)terms.randomKeys->Handle();
			vals = terms.randomVals ? (void*)terms.randomVals->Handle() : 0;
		}
		CUDPPResult result = cudppSort(plan->getHandle(), keys, vals, 
			terms.count);

		if(CUDPP_SUCCESS != result) return CUDA_ERROR_LAUNCH_FAILED;
	}


	*elapsed = timer.Stop();
	return CUDA_SUCCESS;
}

