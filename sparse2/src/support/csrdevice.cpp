#include "sparse.h"

template<typename T>
CUresult CreateCSRDevice(const CSRMatrix<T>& csr, CuContext* context,
	std::auto_ptr<CSRDevice>* ppDevice) {

	std::auto_ptr<CSRDevice> device(new CSRDevice);
	device->height = csr.height;
	device->width = csr.width;
	if(IsSameType<T, float>::value) device->prec = SPARSE_PREC_REAL4;
	else if(IsSameType<T, double>::value) device->prec = SPARSE_PREC_REAL8;
	else if(IsSameType<T, cfloat>::value) device->prec = SPARSE_PREC_COMPLEX4;
	else if(IsSameType<T, cdouble>::value) device->prec = SPARSE_PREC_COMPLEX8;
	
	CUresult result = context->MemAlloc(csr.colIndices, &device->colIndices);
	if(CUDA_SUCCESS != result) return result;

	result = context->MemAlloc(csr.rowIndices, &device->rowIndices);
	if(CUDA_SUCCESS != result) return result;

	result = context->MemAlloc(csr.sparseValues, &device->sparseValues);
	if(CUDA_SUCCESS != result) return result;

	*ppDevice = device;
	return CUDA_SUCCESS;
}


template<typename T>
CUresult CreateCSRDevice(const SparseMatrix<T>& matrix, CuContext* context, 
	std::auto_ptr<CSRDevice>* ppDevice) {
	
	std::auto_ptr<CSRMatrix<T> > csr;
	BuildCSR(matrix, &csr);
	return CreateCSRDevice(*csr, context, ppDevice);
}


template CUresult CreateCSRDevice(const SparseMatrix<float>& matrix, 
	CuContext* context, std::auto_ptr<CSRDevice>* ppDevice);
template CUresult CreateCSRDevice(const SparseMatrix<double>& matrix, 
	CuContext* context, std::auto_ptr<CSRDevice>* ppDevice);
template CUresult CreateCSRDevice(const SparseMatrix<cfloat>& matrix, 
	CuContext* context, std::auto_ptr<CSRDevice>* ppDevice);
template CUresult CreateCSRDevice(const SparseMatrix<cdouble>& matrix, 
	CuContext* context, std::auto_ptr<CSRDevice>* ppDevice);

