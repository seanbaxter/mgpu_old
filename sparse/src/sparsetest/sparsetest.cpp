#include "../../../inc/mgpusparse.h"
#include "../../../util/cucpp.h"
#include "../support/sparse.h"
#include <cusparse.h>
#include <cstdio>
#include <sstream>

const char* Matrices[][2] = {
	{ "scircuit.mtx", "Circuit" },
	{ "pdb1HYS.mtx", "Protein" },
	{ "cant.mtx", "FEM/Cantilever" },
	{ "consph.mtx", "FEM/Spheres" },
	{ "pwtk.mtx", "Wind Tunnel" },
	{ "rma10.mtx", "FEM/Harbor" },
	{ "qcd5_4.mtx", "QCD" },
	{ "shipsec1.mtx", "FEM/Ship" },
	{ "mac_econ_fwd500.mtx", "Economics" },
	{ "mc2depi.mtx", "Epidemiology" },
	{ "cop20k_A.mtx", "FEM/Accelerator" },
	{ "webbase-1M.mtx", "Webbase" }
};


////////////////////////////////////////////////////////////////////////////////
// Sparse matrix benchmarking utilities

const int BytesPerElement[6] = {
	12,		// index (4), float (4), float (4)
	16,		// index (4), float (4), double (8)
	20,		// index (4), double (8), double (8)
	20,		// index (4), cfloat (8), cfloat (8)
	28,		// index (4), cfloat (8), cdouble (16)
	36		// index (4), cdouble (16), cdouble (16)
};

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


bool VerifyMultiplication(const SparseMatrix<double>& matrix, CuDeviceMem* xVec,
	CuDeviceMem* yVec, sparsePrec_t prec) {

	bool result;
	if((SPARSE_PREC_REAL8 == prec) || (SPARSE_PREC_REAL_MIXED == prec)) {
		std::vector<double> xVecHost, yVecHost;
		xVec->ToHost(xVecHost);
		yVec->ToHost(yVecHost);
		result = VerifySparseProduct(matrix, xVecHost, yVecHost);
	} else {
		std::auto_ptr<SparseMatrix<float> > matrix2;
		CopySparseMatrix(matrix, matrix.order, &matrix2);
		std::vector<float> xVecHost, yVecHost;
		xVec->ToHost(xVecHost);
		yVec->ToHost(yVecHost);
		result = VerifySparseProduct(*matrix2, xVecHost, yVecHost);
	}
	return result;
}


////////////////////////////////////////////////////////////////////////////////
// CUSPARSE benchmarking

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
		cuCtxSynchronize();

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


////////////////////////////////////////////////////////////////////////////////
// MGPU Sparse benchmarking

sparseStatus_t BenchmarkMGPUSparse(const SparseMatrix<double>& m, 
	sparseEngine_t mgpu, sparsePrec_t prec, int valuesPerThread, 
	CuDeviceMem* xVec, CuDeviceMem* yVec, int iterations, int numRuns, 
	double* inflation, Benchmark* benchmark) {

	// Encode the sparse matrix
	sparseMat_t mat = 0;
	sparseStatus_t status;

	if(SPARSE_PREC_REAL4 == prec || SPARSE_PREC_REAL_MIXED == prec) {
		std::auto_ptr<SparseMatrix<float> > floatMatrix;
		CopySparseMatrix(m, SparseOrderRow, &floatMatrix);
		status = sparseMatCreateInterleave(mgpu, m.height, m.width, prec, 
			valuesPerThread, SPARSE_INPUT_COO, (int)m.elements.size(),
			&floatMatrix->elements[0], 0, &mat);
	} else
		status = sparseMatCreateInterleave(mgpu, m.height, m.width, prec, 
			valuesPerThread, SPARSE_INPUT_COO, (int)m.elements.size(),
			&m.elements[0], 0, &mat);

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


////////////////////////////////////////////////////////////////////////////////
// Benchmark loop

struct Result {
	const char* matrixName;
	Benchmark cusparse;
	Benchmark mgpu[5];
	Benchmark mgpuPreferred;
};

bool RunBenchmark(CuContext* context, sparseEngine_t mgpu, 
	cusparseHandle_t cusparse, const char* matrixPath, int numRuns,
	int numIterations, int numValueSets, sparsePrec_t prec, bool verify,
	std::vector<Result>& results) {

	printf("Running %d tests with %d iterations.\n\n", numRuns,
		numIterations);
	
	const int NumMatrices = sizeof(Matrices) / sizeof(*Matrices);
	results.resize(NumMatrices);

	cusparseMatDescr_t cusparseMat;
	cusparseCreateMatDescr(&cusparseMat);

	std::vector<std::pair<std::string, double> > benchmarkRatios;
	for(int i(0); i < NumMatrices; ++i) {
		
		// Load the matrix
		std::string err;
		std::string matrixFilename = std::string(matrixPath) + Matrices[i][0];

		printf("Loading matrix %s (%s)\n", Matrices[i][0], Matrices[i][1]);
	
		std::auto_ptr<SparseMatrix<double> > m;
		bool success = ReadSparseMatrix(matrixFilename.c_str(), SparseOrderRow, 
			&m, err);

		if(!success) {
			printf("Error loading file %\n", Matrices[i][0]);
			return false;
		}

		double rowStddev = RowDensityStddev(*m);
		printf("h = %6d, w = %6d, nz = %8d, rowStddev = %7.3lf\n", m->height,
			m->width, m->elements.size(), rowStddev);

		// Allocate the input and output vectors
		DeviceMemPtr xVecDevice, yVecDevice;
		if(SPARSE_PREC_REAL4 == prec) {
			std::vector<float> xVec(m->width);
			for(size_t i(0); i < xVec.size(); ++i)
				xVec[i] = (float)(1 + i);
			context->MemAlloc(xVec, &xVecDevice);
			context->MemAlloc<float>(m->height, &yVecDevice);
		} else {
			std::vector<double> xVec(m->width);
			for(size_t i(0); i < xVec.size(); ++i)
				xVec[i] = (double)(1 + i);
			context->MemAlloc(xVec, &xVecDevice);
			context->MemAlloc<double>(m->height, &yVecDevice);
		}

		std::vector<std::vector<Benchmark> > mgpuBenchmarks(numValueSets);
		for(int j(0); j < numValueSets; ++j) mgpuBenchmarks[j].resize(numRuns);
		std::vector<Benchmark> cusparseBenchmarks(numRuns);

		std::vector<int> mgpuFastest(numValueSets);
		int mgpuFastestRun = 0;


		////////////////////////////////////////////////////////////////////////
		// Benchmark MGPU Sparse

		for(int val(0); val < numValueSets; ++val) {
			double inflation;
			int valsPerThread = 4 * (val + 1);
			BenchmarkMGPUSparse(*m, mgpu, prec, valsPerThread, xVecDevice, 
				yVecDevice, numIterations, numRuns, &inflation, 
				&mgpuBenchmarks[val][0]);

			if(verify) {
				bool match = VerifyMultiplication(*m, xVecDevice, yVecDevice,
					prec);
				if(!match) {
					printf("Multiplication verification FAILED\n\n");
					return false;
				}
			}

			printf("MGPU value size: %d:\n", valsPerThread);

			for(int run(0); run < numRuns; ++run) {
				printf("\t%7.3lf GB/s", mgpuBenchmarks[val][run].bandwidth);
				results[i].mgpu[val].Max(mgpuBenchmarks[val][run]);
			}

			results[i].mgpuPreferred.Max(results[i].mgpu[val]);
		}

		
		////////////////////////////////////////////////////////////////////////
		// Benchmark CUSPARSE

		BenchmarkCuSparse(*m, cusparse, cusparseMat, context, prec,
			xVecDevice, yVecDevice, numIterations, numRuns,
			&cusparseBenchmarks[0]);

		if(verify) {
			bool match = VerifyMultiplication(*m, xVecDevice, yVecDevice, prec);
			if(!match) {
				printf("Multiplication verification FAILED\n\n");
				return false;
			}
		}

		printf("CUSPARSE:\n");
		for(int run(0); run < numRuns; ++run) {
			printf("\t%7.3lf GB/s\n", cusparseBenchmarks[run].bandwidth);
			results[i].cusparse.Max(cusparseBenchmarks[run]);
		}
	}

	cusparseDestroyMatDescr(cusparseMat);

	return true;
}



////////////////////////////////////////////////////////////////////////////////
// sparsetest main

int main(int argc, char** argv) {

	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);

	sparseEngine_t mgpu;
	sparseStatus_t status = sparseCreate("../../src/cubin/", &mgpu);

	cusparseHandle_t cusparse;
	cusparseStatus_t cuStatus = cusparseCreate(&cusparse);

	std::vector<Result> results;

	RunBenchmark(context, mgpu, cusparse, "../../matrices/", 3, 100, 5, 
		SPARSE_PREC_REAL8, true, results);

	
	cusparseDestroy(cusparse);

	sparseRelease(mgpu);
}

