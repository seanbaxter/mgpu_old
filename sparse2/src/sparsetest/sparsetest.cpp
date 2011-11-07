#include "benchmark.h"

#include <cstdio>
#include <sstream>

const char* Matrices[][2] = {
	{ "scircuit.mtx", "Circuit" },
	{ "mac_econ_fwd500.mtx", "Economics" },
	{ "mc2depi.mtx", "Epidemiology" },
	{ "cop20k_A.mtx", "FEM/Accelerator" },
	{ "cant.mtx", "FEM/Cantilever" },
	{ "rma10.mtx", "FEM/Harbor" },
	{ "shipsec1.mtx", "FEM/Ship" },
	{ "consph.mtx", "FEM/Spheres" },
	{ "pdb1HYS.mtx", "Protein" },
	{ "qcd5_4.mtx", "QCD" },
	{ "webbase-1M.mtx", "Webbase" },
	{ "pwtk.mtx", "Wind Tunnel" }
};

const int BytesPerElement[6] = {
	12,		// index (4), float (4), float (g4)
	20,		// index (4), double (8), double (8)
	20,		// index (4), cfloat (8), cfloat (8)
	36		// index (4), cdouble (16), cdouble (16)
};

const int NumMgpuSizes = 7;
const int MgpuSizes[NumMgpuSizes] = {
	4, 6, 8, 10, 12, 16, 20
};
;
struct Result {
	const char* matrixName;
	Benchmark mgpu[5];
	Benchmark mgpuPreferred;

	Benchmark cusparse;
	Benchmark cuspCsr;
	Benchmark cuspEll;
};


bool VerifyMultiplication(const SparseMatrix<double>& matrix, CuDeviceMem* xVec,
	CuDeviceMem* yVec, sparsePrec_t prec) {

	bool result;
	if(SPARSE_PREC_REAL8 == prec) {
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
	CUresult result;

	std::vector<std::pair<std::string, double> > benchmarkRatios;
	for(int i(0); i < NumMatrices; ++i) {

		////////////////////////////////////////////////////////////////////////
		// Load the matrix

		std::string err;
		std::string matrixFilename = std::string(matrixPath) + Matrices[i][0];

		printf("Loading matrix %s (%s)\n", Matrices[i][0], Matrices[i][1]);

		std::auto_ptr<SparseMatrix<double> > m;
		bool success = ReadSparseMatrix(matrixFilename.c_str(), SparseOrderRow, 
			&m, err);

		if(!success) {
			printf("Error loading file %s\n", Matrices[i][0]);
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

		std::vector<Benchmark> cuspCsrBenchmarks(numRuns);

		std::vector<Benchmark> cuspEllBenchmarks(numRuns);

		std::vector<int> mgpuFastest(numValueSets);


		////////////////////////////////////////////////////////////////////////
		// Benchmark MGPU Sparse

		results[i].matrixName = Matrices[i][1];

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
				printf("\t%7.3lf GB/s\n", mgpuBenchmarks[val][run].bandwidth);
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


		////////////////////////////////////////////////////////////////////////
		// Benchmark CUSP CSR

		result = BenchmarkCusp(*m, EncodingCsr, prec, xVecDevice, yVecDevice, 
			numIterations, numRuns, &cuspCsrBenchmarks[0]);

		if(verify) {
			bool match = VerifyMultiplication(*m, xVecDevice, yVecDevice, prec);
			if(!match) {
				printf("Multiplication verification FAILED\n\n");
				return false;
			}
		}
		printf("CUSP CSR:\n");
		for(int run(0); run < numRuns; ++run) {
			printf("\t%7.3lf GB/s\n", cuspCsrBenchmarks[run].bandwidth);
			results[i].cuspCsr.Max(cuspCsrBenchmarks[run]);
		}


		////////////////////////////////////////////////////////////////////////
		// Benchmark CUSP ELLPACK

		result = BenchmarkCusp(*m, EncodingEll, prec, xVecDevice, yVecDevice, 
			numIterations, numRuns, &cuspEllBenchmarks[0]);
		if(CUDA_ERROR_OUT_OF_MEMORY == result) {
			printf("ELLPACK is OUT OF MEMORY.\n");
			continue;
		}

		if(verify) {
			bool match = VerifyMultiplication(*m, xVecDevice, yVecDevice, prec);
			if(!match) {
				printf("Multiplication verification FAILED\n\n");
				return false;
			}
		}
		printf("CUSP ELLPACK:\n");
		for(int run(0); run < numRuns; ++run) {
			printf("\t%7.3lf GB/s\n", cuspEllBenchmarks[run].bandwidth);
			results[i].cuspEll.Max(cuspEllBenchmarks[run]);
		}
	}

	cusparseDestroyMatDescr(cusparseMat);

	return true;
}

int main(int argc, char** argv) {

	cuInit(0);

	DevicePtr device;
	CreateCuDevice(0, &device);

	ContextPtr context;
	CreateCuContext(device, 0, &context);
//	CUresult result = AttachCuContext(&context);

	sparseEngine_t mgpu;
	sparseStatus_t status = sparseCreate("../../src/cubin/", &mgpu);

	if(SPARSE_STATUS_SUCCESS != status) {
		printf("Could not create MGPU Sparse object: %s.\n",
			sparseStatusString(status));
		return 0;
	}

	cusparseHandle_t cusparse;
	cusparseStatus_t cuStatus = cusparseCreate(&cusparse);
	if(CUSPARSE_STATUS_SUCCESS != cuStatus) {
		printf("Could not create CUSPARSE object.\n");
		return 0;
	}

	std::vector<Result> results;

	RunBenchmark(context, mgpu, cusparse, "../../../sparse/matrices/", 3, 400, 
		5, SPARSE_PREC_REAL4, true, results);

	printf("\n\n");
	printf("         MATRIX NAME             MGPU         CUSPARSE      CUSP-CSR       CUSP-ELL\n");
	for(size_t i(0); i < results.size(); ++i) {
		printf("%20s     %7.3lf GB/s     %7.3lf GB/s  %7.3lf GB/s   %7.3lf GB/s\n",
			results[i].matrixName, results[i].mgpuPreferred.bandwidth, results[i].cusparse.bandwidth,
			results[i].cuspCsr.bandwidth, results[i].cuspEll.bandwidth);
	}

	cusparseDestroy(cusparse);

	sparseRelease(mgpu);
}
