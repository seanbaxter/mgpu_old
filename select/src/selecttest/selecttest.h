#pragma once

#include "../../../util/cucpp.h"
#include "../../../inc/mgpuselect.h"
#include "../../../inc/mgpusort.hpp"

// MGPU Select
selectStatus_t SelectBenchmark(int iterations, int count, int k, 
	CuContext* context, selectEngine_t engine, CuDeviceMem* randomKeys, 
	selectType_t type, double* elapsed, uint* element);

// MGPU Sort
sortStatus_t MgpuSortBenchmark(int iterations, int count, int k, 
	CuContext* context, sortEngine_t engine, CuDeviceMem* randomKeys,
	double* elapsed, uint* element);

// thrust::sort
CUresult ThrustBenchmark(int iterations, int count, int k, CuContext* context,
	CuDeviceMem* randomKeys, selectType_t type, double* elapsed, uint* element);

