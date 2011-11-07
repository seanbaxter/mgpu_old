#include "benchmark.h"

cusparseStatus_t BenchmarkCuSparse(const SparseMatrix<double>& m,
	cusparseHandle_t cusparse, cusparseMatDescr_t desc, CuContext* context,
	sparsePrec_t prec, CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations, 
	int numRuns, Benchmark* benchmark) {

	// Encode the sparse matrix
	std::auto_ptr<CSRDevice> csr;
	CUresult result;

	if(SPARSE_PREC_REAL4 == prec) {
		std::auto_ptr<SparseMatrix<float> > floatMatrix;
		CopySparseMatrix(m, SparseOrderRow, &floatMatrix);
		result = CreateCSRDevice(*floatMatrix, context, &csr);
	} else
		result = CreateCSRDevice(m, context, &csr);

	if(CUDA_SUCCESS != result) return CUSPARSE_STATUS_EXECUTION_FAILED;


	for(int curRun(0); curRun < numRuns; ++curRun) {
		CuEventTimer timer;
		timer.Start();

		for(int it(0); it < iterations; ++it) {
			cusparseStatus_t status;
			if(SPARSE_PREC_REAL4 == prec)
				status = cusparseScsrmv(cusparse, 
					CUSPARSE_OPERATION_NON_TRANSPOSE, m.height, m.width, 1.0f,
					desc, (const float*)csr->sparseValues->Handle(),
					(const int*)csr->rowIndices->Handle(), 
					(const int*)csr->colIndices->Handle(),
					(float*)xVec->Handle(), 0.0f, (float*)yVec->Handle());
			else
				status = cusparseDcsrmv(cusparse,
					CUSPARSE_OPERATION_NON_TRANSPOSE, m.height, m.width, 1.0,
					desc, (const double*)csr->sparseValues->Handle(),
					(const int*)csr->rowIndices->Handle(), 
					(const int*)csr->colIndices->Handle(),
					(double*)xVec->Handle(), 0.0, (double*)yVec->Handle());

			if(CUSPARSE_STATUS_SUCCESS != status) return status;
		}

		double elapsed = timer.Stop();

		benchmark[curRun].Calculate(elapsed, iterations, prec, m);
	}

	return CUSPARSE_STATUS_SUCCESS;
}