// mm/sparse.h wraps the Matrix Market mmio functions to allow easy access to
// sparse matrix files in this format.

#pragma once

#include <vector>
#include "../../../inc/mgpusparse.h"
#include "../../../util/cucpp.h"

typedef unsigned int uint;
typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;

typedef sparseMatElementReal8COO_t Element;


////////////////////////////////////////////////////////////////////////////////
// Use Matrix Market library for reading in sparse matrices.
// This is the NVIDIA matrix test format

// Sparse matrix elements returned by ReadSparseMatrix.

enum SparseOrder {
	SparseOrderNone,
	SparseOrderRow,
	SparseOrderCol
};

template<typename T>
struct SparseMatrix {
	std::string filename;
	int height, width;
	SparseOrder order;
	
	std::vector<sparseMatElementCOO_t<T> > elements;

	static inline bool SparseOrderColPred(const sparseMatElementCOO_t<T>& a,
		const sparseMatElementCOO_t<T>& b) {

		if(a.col < b.col) return true;
		if(b.col < a.col) return false;
		return a.row < b.row;
	}

	void ChangeOrder(SparseOrder order2) {
		if(order != order2) {
			if(SparseOrderRow == order2) 
#ifdef _DEBUG
				qsort(&elements[0], elements.size(), 
					sizeof(sparseMatElementCOO_t<T>), 
					&sparseMatElementCOO_t<T>::Cmp);
#else
				std::sort(elements.begin(), elements.end());
#endif
			else if(SparseOrderCol == order2)
				std::sort(elements.begin(), elements.end(), SparseOrderColPred);
			order = order2;
		} 
	}
};


template<typename T, typename T2>
inline void CopySparseMatrix(const SparseMatrix<T>& source, SparseOrder order,
	std::auto_ptr<SparseMatrix<T2> >* ppDest) {
	std::auto_ptr<SparseMatrix<T2> > dest(new SparseMatrix<T2>);
	dest->filename = source.filename;
	dest->height = source.height;
	dest->width = source.width;

	dest->elements.resize(source.elements.size());
	for(size_t i(0); i < dest->elements.size(); ++i) {
		sparseMatElementCOO_t<T> e = source.elements[i];
		sparseMatElementCOO_t<T2>& e2 = dest->elements[i];
		e2.value = (T2)e.value;
		e2.row = e.row;
		e2.col = e.col;
	}

	dest->order = source.order;
	dest->ChangeOrder(order);

	*ppDest = dest;
}


// Read a matrix in Matrix Market format.  Requires real matrices with values.
bool ReadSparseMatrix(const char* filename, SparseOrder order,
	std::auto_ptr<SparseMatrix<double> >* ppSparse, std::string& err);

bool WriteSparseMatrix(const char* filename, 
	const SparseMatrix<double>& matrix);

template<typename T>
void ComputeSparseProduct(const SparseMatrix<T>& matrix, 
	const std::vector<T>& xVec, std::vector<T>& yVec, 
	std::vector<T>* absSums = 0);

template<typename T>
bool CompareSparseProducts(const std::vector<T>& a, const std::vector<T>& b,
	const std::vector<T>& absSums);

template<typename T>
bool VerifySparseProduct(const SparseMatrix<T>& matrix, 
	const std::vector<T>& xVec, const std::vector<T>& yVec);

double RowDensityStddev(const SparseMatrix<double>& matrix);


template<typename T>
void DeinterleaveMatrix(const SparseMatrix<T>& matrix, 
	std::vector<int>& rowIndices, std::vector<int>& colIndices, 
	std::vector<T>& sparseValues) {

	size_t size = matrix.elements.size();
	rowIndices.resize(size);
	colIndices.resize(size);
	sparseValues.resize(size);
	for(size_t i(0); i < size; ++i) {
		rowIndices[i] = matrix.elements[i].row;
		colIndices[i] = matrix.elements[i].col;
		sparseValues[i] = matrix.elements[i].value;
	}
}


////////////////////////////////////////////////////////////////////////////////
// CSRMatrix
// Compressed Sparse Row format as described in CUSPARSE_Library.pdf

template<typename T>
struct CSRMatrix {
	int height, width;
	std::vector<int> colIndices, rowIndices;
	std::vector<T> sparseValues;
};

template<typename T>
void BuildCSR(const SparseMatrix<T>& matrix, 
	std::auto_ptr<CSRMatrix<T> >* ppCSR);


////////////////////////////////////////////////////////////////////////////////
// CSCMatrix

template<typename T>
struct CSCMatrix {
	int height, width;
	std::vector<int> rowIndices, colIndices;
	std::vector<T> sparseValues;
};

template<typename T>
void BuildCSC(const SparseMatrix<T>& matrix, 
	std::auto_ptr<CSCMatrix<T> >* ppCSC);


////////////////////////////////////////////////////////////////////////////////
// Compressed Sparse Row encoding - used by CUSPARSE

struct CSRDevice {
	int height, width;
	DeviceMemPtr rowIndices, colIndices, sparseValues;
	sparsePrec_t prec;
};

template<typename T>
CUresult CreateCSRDevice(const CSRMatrix<T>& csr, CuContext* context,
	std::auto_ptr<CSRDevice>* ppDevice);

template<typename T>
CUresult CreateCSRDevice(const SparseMatrix<T>& matrix, CuContext* context, 
	std::auto_ptr<CSRDevice>* ppDevice);
