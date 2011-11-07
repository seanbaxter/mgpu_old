#include "benchmark.h"
#include <cusp/csr_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>

template<typename M, typename T>
void BenchmarkCusp3(M& m, const SparseMatrix<T>& m2, 
	cusp::array1d<T, cusp::device_memory>& xVec, 
	cusp::array1d<T, cusp::device_memory>& yVec, sparsePrec_t prec,
	int iterations, int numRuns, Benchmark* benchmark) {

	for(int curRun(0); curRun < numRuns; ++curRun) {
		CuEventTimer timer;
		timer.Start();

		for(int it(0); it < iterations; ++it)
			cusp::multiply(m, xVec, yVec);

		double elapsed = timer.Stop();

		benchmark[curRun].Calculate(elapsed, iterations, prec, m2);
	}
}

template<typename T>
CUresult BenchmarkCusp2(const SparseMatrix<T>& m, Encoding encoding,
	sparsePrec_t prec, CuDeviceMem* xVec, CuDeviceMem* yVec,
	int iterations, int numRuns, Benchmark* benchmark) {

	std::auto_ptr<CSRMatrix<T> > csrHost;
	BuildCSR(m, &csrHost);

	// Create the CUSP CSR matrix and copy the values from host memory.
	typedef cusp::csr_matrix<int, T, cusp::device_memory> Csr;
	typedef cusp::ell_matrix<int, T, cusp::device_memory> Ell;
	Csr csr(m.height, m.width, m.elements.size());
	thrust::copy(csrHost->rowIndices.begin(), csrHost->rowIndices.end(),
		csr.row_offsets.begin());
	thrust::copy(csrHost->colIndices.begin(), csrHost->colIndices.end(),
		csr.column_indices.begin());
	thrust::copy(csrHost->sparseValues.begin(), csrHost->sparseValues.end(),
		csr.values.begin());

	// Copy the xVec values to a cusp::array1d structure.
	cusp::array1d<T, cusp::device_memory> xArray(m.width),
		yArray(m.height);
	thrust::copy(thrust::device_ptr<T>((T*)xVec->Handle()),
		thrust::device_ptr<T>((T*)xVec->Handle()) + m.width,
		xArray.begin());

	if(EncodingEll == encoding) {
		Ell ell;
		try {
			cusp::detail::device::convert(csr, ell, Csr::format(), 
				Ell::format(), 100000.0f);
		} catch(...) {
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		BenchmarkCusp3(ell, m, xArray, yArray, prec, iterations, numRuns, 
			benchmark);
	} else if(EncodingCsr == encoding)
		BenchmarkCusp3(csr, m, xArray, yArray, prec, iterations, numRuns, 
			benchmark);

	// copy yArray back to yVec.
	thrust::copy(yArray.begin(), yArray.end(), 
		thrust::device_ptr<T>((T*)yVec->Handle()));

	return CUDA_SUCCESS;
}


CUresult BenchmarkCusp(const SparseMatrix<double>& m, Encoding encoding,
	sparsePrec_t prec, CuDeviceMem* xVec, CuDeviceMem* yVec,
	int iterations, int numRuns, Benchmark* benchmark) {

	CUresult result;
	if(SPARSE_PREC_REAL4 == prec) {
		std::auto_ptr<SparseMatrix<float> > m2;
		CopySparseMatrix(m, SparseOrderRow, &m2);
		result = BenchmarkCusp2<float>(*m2, encoding, prec, xVec, yVec, 
			iterations, numRuns, benchmark);
	} else if(SPARSE_PREC_REAL8 == prec) {
		result = BenchmarkCusp2<double>(m, encoding, prec, xVec, yVec, 
			iterations, numRuns, benchmark);
	}
	return result;
}
