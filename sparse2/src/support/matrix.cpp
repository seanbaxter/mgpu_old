#include "sparse.h"
#include "mmio.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cassert>


////////////////////////////////////////////////////////////////////////////////
// ReadSparseMatrix

bool ReadSparse2(FILE* f, SparseMatrix<double>& sparse, std::string& err) {

	MM_typecode matcode;
	int code = mm_read_banner(f, &matcode);
	if(code) {
		err = "Not a sparse matrix.\n";
		return false;
	}

	bool isPattern = mm_is_pattern(matcode);
	bool isReal = mm_is_real(matcode);

	if(!(isPattern || isReal) || !mm_is_matrix(matcode) || 
		!mm_is_coordinate(matcode) || !mm_is_sparse(matcode)) {
		err = "Not a real matrix.\n";
		return false;
	}
	
	int nz;
	int height, width;
	code = mm_read_mtx_crd_size(f, &height, &width, &nz);
	std::vector<sparseMatElementReal8COO_t> elements;
	elements.reserve(nz);

	for(int i(0); i < nz; ++i) {
		sparseMatElementReal8COO_t e;
		int x, y;

		if(isReal) {
			int count = fscanf(f, "%d %d %lf", &y, &x, &e.value);
			if(3 != count) {
				std::ostringstream oss;
				oss<< "Error parsing line "<< (i + 1);
				err = oss.str();
				return false;
			}
		} else if(isPattern) {
			int count = fscanf(f, "%d %df", &y, &x);
			if(2 != count) {
				std::ostringstream oss;
				oss<< "Error parsing line "<< (i + 1);
				err = oss.str();
				return false;
			}
			e.value = 1;
		}
		e.row = y - 1;
		e.col = x - 1;
		elements.push_back(e);

		if((mm_is_symmetric(matcode) || mm_is_skew(matcode)) && (x != y)) {
			std::swap(e.row, e.col);
			if(mm_is_skew(matcode)) e.value = -e.value;
			elements.push_back(e);
		}
	}

	sparse.height = height;
	sparse.width = width;
	sparse.elements.swap(elements);

	return true;
}

bool ReadSparseMatrix(const char* filename, SparseOrder order, 
	std::auto_ptr<SparseMatrix<double> >* ppSparse, std::string& err) {

	FILE* f = fopen(filename, "r");
	if(!f) {
		err = "Could not open file for reading";
		return false;
	}

	std::auto_ptr<SparseMatrix<double> > sparse(new SparseMatrix<double>);
	bool result = ReadSparse2(f, *sparse, err);

	fclose(f);

	if(!result) return false;

	sparse->order = SparseOrderNone;
	sparse->filename = filename;
	sparse->ChangeOrder(order);
	*ppSparse = sparse;
	return result;
}


////////////////////////////////////////////////////////////////////////////////
// WriteSparseMatrix

bool WriteSparseMatrix(const char* filename, 
	const SparseMatrix<double>& matrix) {
	
	// Use the high level routine mm_write_mtx_crd
	// int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], 
	//		int J[], double val[], MM_typecode matcode);

	int nz = (int)matrix.elements.size();
	std::vector<int> I(nz), J(nz);
	std::vector<double> val(nz);

	for(int i(0); i < nz; ++i) {
		// Matrix Market format is 1-indexed.
		I[i] = matrix.elements[i].row + 1;
		J[i] = matrix.elements[i].col + 1;
		val[i] = matrix.elements[i].value;
	}

	MM_typecode typecode;
	mm_initialize_typecode(&typecode);
	mm_set_matrix(&typecode);
	mm_set_sparse(&typecode);
	mm_set_real(&typecode);
	mm_set_general(&typecode);

	int status = mm_write_mtx_crd((char*)filename, matrix.height, matrix.width,
		nz, &I[0], &J[0], &val[0], typecode);
	return 0 == status;
}


////////////////////////////////////////////////////////////////////////////////
// ComputeSparseProduct

template<typename T>
void ComputeSparseProduct(const SparseMatrix<T>& matrix, 
	const std::vector<T>& xVec, std::vector<T>& yVec, std::vector<T>* absSums) {

	yVec.resize(0);
	yVec.resize(matrix.height);

	if(absSums) {
		absSums->resize(0);
		absSums->resize(matrix.height);
		for(size_t i(0); i < matrix.elements.size(); ++i) {
			sparseMatElementCOO_t<T> e = matrix.elements[i];
			T product = xVec[e.col] * e.value;
			yVec[e.row] += product;
			(*absSums)[e.row] += std::abs(product);
		}
	} else
		for(size_t i(0); i < matrix.elements.size(); ++i) {
			sparseMatElementCOO_t<T> e = matrix.elements[i];
			yVec[e.row] += xVec[e.col] * e.value;
		}
}

template void ComputeSparseProduct(const SparseMatrix<float>& matrix, 
	const std::vector<float>& xVec, std::vector<float>& yVec,
	std::vector<float>* absSums);
template void ComputeSparseProduct(const SparseMatrix<double>& matrix,
	const std::vector<double>& xVec, std::vector<double>& yVec,
	std::vector<double>* absSums);


////////////////////////////////////////////////////////////////////////////////
// CompareSparseProducts

template<typename T>
bool CompareSparseProducts(const std::vector<T>& a, const std::vector<T>& b,
	const std::vector<T>& absSums) {

	assert(a.size() == b.size() && a.size() == absSums.size());
	for(size_t i(0); i < a.size(); ++i) {
		T diff = a[i] - b[i];
		if(8 == sizeof(T)) {
			// double precision - use 1.0e-8 for diff comparison and 1.0e-9
			// for magnitude comparison
			if(std::abs(diff) > 1.e-8 && std::abs(diff) > (1.e-9 * absSums[i]))
				return false;
		} else {
			// double precision - use 1.0e-5 for diff comparison and 1.0e-5
			// for magnitude comparison
			if(std::abs(diff) > 1.e-4 && std::abs(diff) > (1.e-5 * absSums[i])) 
				return false;
		}
	}
	return true;
}

template bool CompareSparseProducts(const std::vector<float>& a,
	const std::vector<float>& b, const std::vector<float>& absSums);
template bool CompareSparseProducts(const std::vector<double>& a,
	const std::vector<double>& b, const std::vector<double>& absSums);


////////////////////////////////////////////////////////////////////////////////
// VerifySparseProduct

template<typename T>
bool VerifySparseProduct(const SparseMatrix<T>& matrix,
	const std::vector<T>& xVec, const std::vector<T>& yVec) {

	std::vector<T> yVecRef, absSums;
	ComputeSparseProduct(matrix, xVec, yVecRef, &absSums);
	return CompareSparseProducts(yVecRef, yVec, absSums);
}

template bool VerifySparseProduct(const SparseMatrix<float>& matrix,
	const std::vector<float>& xVec, const std::vector<float>& yVec);
template bool VerifySparseProduct(const SparseMatrix<double>& matrix,
	const std::vector<double>& xVec, const std::vector<double>& yVec);


////////////////////////////////////////////////////////////////////////////////
// RowDensityStddev

double RowDensityStddev(const SparseMatrix<double>& matrix, int* maxLen) {
	double meanNzPerRow = (double)matrix.elements.size() / matrix.height;
	double squaredSum = 0;

	if(maxLen) *maxLen = 0;

	int nz = matrix.elements.size();
	int prevRow = 0;
	int curRowCount = 0;
	for(int i(0); i < nz; ) {
		int row = matrix.elements[i].row;
		if(row == prevRow) {
			++curRowCount;
			++i;
		} else {
			squaredSum += (curRowCount - meanNzPerRow) * 
				(curRowCount - meanNzPerRow);
			if(maxLen) *maxLen = std::max(*maxLen, curRowCount);
			curRowCount = 0;
			++prevRow;
		}
	}

	return sqrt(squaredSum / matrix.height);
}


////////////////////////////////////////////////////////////////////////////////
// BuildCSR

template<typename T>
void BuildCSR(const SparseMatrix<T>& matrix, 
	std::auto_ptr<CSRMatrix<T> >* ppCSR) {

	// If the data is not sorted by row, do that
	std::auto_ptr<SparseMatrix<T> > rowSorted;
	const SparseMatrix<T>* m = &matrix;

	if(SparseOrderRow != matrix.order) {
		CopySparseMatrix(matrix, SparseOrderRow, &rowSorted);
		m = rowSorted.get();
	}

	std::auto_ptr<CSRMatrix<T> > csr(new CSRMatrix<T>);
	csr->height = m->height;
	csr->width = m->width;
	csr->rowIndices.resize(m->height + 1);
	csr->colIndices.resize(m->elements.size());
	csr->sparseValues.resize(m->elements.size());

	int curRow(0);
	int i(0);
	const sparseMatElementCOO_t<T>* elements = &m->elements[0];
	int nz = (int)m->elements.size();
	
	while(i < nz) {
		// process each row
		sparseMatElementCOO_t<T> e = elements[i];
		if(e.row == curRow) {
			csr->colIndices[i] = e.col;
			csr->sparseValues[i] = (T)e.value;
			++i;
		} else {
			// cap off the row
			++curRow;
			csr->rowIndices[curRow] = i;
			continue;
		}
	}
	csr->rowIndices.back() = nz;

	*ppCSR = csr;
}

template void BuildCSR(const SparseMatrix<float>& matrix, 
	std::auto_ptr<CSRMatrix<float> >* ppCSR);
template void BuildCSR(const SparseMatrix<double>& matrix, 
	std::auto_ptr<CSRMatrix<double> >* ppCSR);
template void BuildCSR(const SparseMatrix<cfloat>& matrix, 
	std::auto_ptr<CSRMatrix<cfloat> >* ppCSR);
template void BuildCSR(const SparseMatrix<cdouble>& matrix, 
	std::auto_ptr<CSRMatrix<cdouble> >* ppCSR);


////////////////////////////////////////////////////////////////////////////////
// BuildCSC

template<typename T>
void BuildCSC(const SparseMatrix<T>& matrix, 
	std::auto_ptr<CSCMatrix<T> >* ppCSC) {

	// If the data is not sorted by row, do that
	std::auto_ptr<SparseMatrix<T> > rowSorted;
	const SparseMatrix<T>* m = &matrix;

	if(SparseOrderCol != matrix.order) {
		CopySparseMatrix(matrix, SparseOrderRow, &rowSorted);
		m = rowSorted.get();
	}

	std::auto_ptr<CSCMatrix<T> > csc(new CSCMatrix<T>);
	csc->height = m->height;
	csc->width = m->width;
	csc->rowIndices.resize(m->elements.size());
	csc->colIndices.resize(m->width + 1);
	csc->sparseValues.resize(m->elements.size());

	int curCol(0);
	int i(0);
	const sparseMatElementCOO_t<T>* elements = &m->elements[0];
	int nz = (int)m->elements.size();
	
	while(i < nz) {
		// process each col
		sparseMatElementCOO_t<T> e = elements[i];
		if(e.col == curCol) {
			csc->rowIndices[i] = e.row;
			csc->sparseValues[i] = (T)e.value;
			++i;
		} else {
			// cap off the col
			++curCol;
			csc->colIndices[curCol] = i;
			continue;
		}
	}
	csc->colIndices.back() = nz;

	*ppCSC = csc;
}

template void BuildCSC(const SparseMatrix<float>& matrix, 
	std::auto_ptr<CSCMatrix<float> >* ppCSC);
template void BuildCSC(const SparseMatrix<double>& matrix, 
	std::auto_ptr<CSCMatrix<double> >* ppCSC);
