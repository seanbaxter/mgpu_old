#include "benchmark.h"

////////////////////////////////////////////////////////////////////////////////
// MGPU Sparse benchmarking

sparseStatus_t BenchmarkMGPUSparse(const SparseMatrix<double>& m, 
	sparseEngine_t mgpu, sparsePrec_t prec, int valuesPerThread, 
	CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations, int numRuns, 
	double* inflation, Benchmark* benchmark) {

	// Encode the sparse matrix
	sparseMat_t mat = 0;
	sparseStatus_t status;

	if(SPARSE_PREC_REAL4 == prec) {
		std::auto_ptr<SparseMatrix<float> > floatMatrix;
		CopySparseMatrix(m, SparseOrderRow, &floatMatrix);

		std::vector<int> rowIndices, colIndices;
		std::vector<float> sparseValues;
		DeinterleaveMatrix(*floatMatrix, rowIndices, colIndices, sparseValues);

		status = sparseMatCreate(mgpu, m.height, m.width, prec, valuesPerThread,
			SPARSE_INPUT_COO, (int)m.elements.size(), &sparseValues[0],
			&rowIndices[0], &colIndices[0], &mat);
	} else {
		std::vector<int> rowIndices, colIndices;
		std::vector<double> sparseValues;
		DeinterleaveMatrix(m, rowIndices, colIndices, sparseValues);

		status = sparseMatCreate(mgpu, m.height, m.width, prec, valuesPerThread,
			SPARSE_INPUT_COO, (int)m.elements.size(), &sparseValues[0],
			&rowIndices[0], &colIndices[0], &mat);
	}

	if(SPARSE_STATUS_SUCCESS != status) return status;


	*inflation = (double)mat->nz2 / mat->nz;

	for(int curRun(0); curRun < numRuns; ++curRun) {
		CuEventTimer timer;
		timer.Start();

		for(int it(0); it < iterations; ++it) {
			if(SPARSE_PREC_REAL4 == prec)
				status = sparseMatVecSMul(mgpu, 1.0f, mat, xVec->Handle(), 0.0f, 
					yVec->Handle());
			else 
				status = sparseMatVecDMul(mgpu, 1.0, mat, xVec->Handle(), 0.0, 
					yVec->Handle());

			if(SPARSE_STATUS_SUCCESS != status) return status;
		}

		double elapsed = timer.Stop();

		benchmark[curRun].Calculate(elapsed, iterations, prec, m);
	}

	sparseMatDestroy(mat);

	return SPARSE_STATUS_SUCCESS;
}
