#include <cusp/csr_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/multiply.h>
#include <cusp/convert.h>
#include "../support/sparse.h"

typedef cusp::csr_matrix<int, double, cusp::device_memory> CuspCsrFloat;
typedef cusp::csr_matrix<int, double, cusp::device_memory> CuspCsrDouble;

typedef cusp::ell_matrix<int, double, cusp::device_memory> CuspEllFloat;
typedef cusp::ell_matrix<int, double, cusp::device_memory> CuspEllDouble;

typedef thrust::device_ptr<double> DoublePtr;
typedef thrust::device_ptr<float> FloatPtr;

int main(int argc, char** argv) {

	ContextPtr context;
	CUresult result = AttachCuContext(&context);
	
	std::auto_ptr<SparseMatrix<double> > m;
	std::string err;
	bool success = ReadSparseMatrix("../../../sparse/matrices/scircuit.mtx",
		SparseOrderRow, &m, err);
	if(!success) {
		printf("Could not load matrix: %s\n", err.c_str());
		return 0;
	}

	// Convert the matrix to CSR format. This deinterleaves the values.
	std::auto_ptr<CSRMatrix<double> > csrHost;
	BuildCSR(*m, &csrHost);

	CuspCsrDouble csr(m->height, m->width, m->elements.size());

	thrust::copy(csrHost->rowIndices.begin(), csrHost->rowIndices.end(),
		csr.row_offsets.begin());
	
	thrust::copy(csrHost->colIndices.begin(), csrHost->colIndices.end(),
		csr.column_indices.begin());

	thrust::copy(csrHost->sparseValues.begin(), csrHost->sparseValues.end(),
		csr.values.begin());

	std::vector<double> xVecHost(m->width);
	for(int i(0); i < m->width; ++i)
		xVecHost[i] = i + 1;

	cusp::array1d<double, cusp::device_memory> xArray(m->width),
		yArray(m->height);
	thrust::copy(xVecHost.begin(), xVecHost.end(), xArray.begin());

//	cusp::multiply(csr, xArray, yArray);
	
	std::vector<double> yVecHost(m->height);
//	thrust::copy(yArray.begin(), yArray.end(), yVecHost.begin());


	// foo
	CuspEllDouble ell;
	cusp::detail::device::convert(csr, ell, CuspCsrDouble::format(), 
		CuspEllDouble::format(), 100.0f);

	cusp::multiply(csr, xArray, yArray);
	thrust::copy(yArray.begin(), yArray.end(), yVecHost.begin());

	int j = 0;
}

