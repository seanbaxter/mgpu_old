#pragma once

#include "../../../inc/mgpusparse.h"
#include "../../../util/cucpp.h"
#include "../support/sparse.h"

#include <cusparse.h>

struct Benchmark {
	double runningTime;
	double nzPerRow;
	double nzPerSecond;
	double bandwidth;

	Benchmark() { 
		runningTime = nzPerRow = nzPerSecond = bandwidth = 0;
	}

	template<typename T>
	void Calculate(double elapsed, int iterations, sparsePrec_t prec,
		const SparseMatrix<T>& matrix) {

		double nz = matrix.elements.size();
		runningTime = elapsed;
		nzPerRow = nz / matrix.height;
		nzPerSecond = (nz * iterations) / elapsed;

		const int BytesPerElement[] = {
			12, 20, 20, 36
		};
		bandwidth = BytesPerElement[(int)prec] * nzPerSecond / (1<< 30);
	}

	void Max(const Benchmark& rhs) {
		if(rhs.bandwidth > bandwidth) {
			runningTime = rhs.runningTime;
			nzPerRow = rhs.nzPerRow;
			nzPerSecond = rhs.nzPerSecond;
			bandwidth = rhs.bandwidth;
		}
	}
};

enum Encoding {
	EncodingCoo,
	EncodingCsr, 
	EncodingEll
};

sparseStatus_t BenchmarkMGPUSparse(const SparseMatrix<double>& m, 
	sparseEngine_t mgpu, sparsePrec_t prec, int valuesPerThread, 
	CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations, int numRuns, 
	double* inflation, Benchmark* benchmark);

cusparseStatus_t BenchmarkCuSparse(const SparseMatrix<double>& m,
	cusparseHandle_t cusparse, cusparseMatDescr_t desc, CuContext* context,
	sparsePrec_t prec, CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations, 
	int numRuns, Benchmark* benchmark);

CUresult BenchmarkCusp(const SparseMatrix<double>& m, Encoding encoding,
	sparsePrec_t prec, CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations,
	int numRuns, Benchmark* benchmark);
