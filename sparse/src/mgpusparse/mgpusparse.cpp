
#include "engine.h"
#include <vector>


////////////////////////////////////////////////////////////////////////////////
// Create and destroy sparse engines

sparseStatus_t SPARSEAPI sparseCreate(const char* kernelPath, 
	sparseEngine_t* engine) {
	
	EnginePtr engine2;
	sparseStatus_t status = CreateSparseEngine(kernelPath, &engine2);
	if(SPARSE_STATUS_SUCCESS == status) *engine = engine2.release();
	return status;
}

sparseStatus_t SPARSEAPI sparseInc(sparseEngine_t engine) {
	engine->AddRef();
	return SPARSE_STATUS_SUCCESS;
}

sparseStatus_t SPARSEAPI sparseDestroy(sparseEngine_t engine) {
	engine->Release();
	return SPARSE_STATUS_SUCCESS;
}

sparseStatus_t SPARSEAPI sparseQuerySupport(sparseEngine_t engine,
	sparsePrec_t prec, int valuesPerThread) {

	sparseEngine_d::Kernel* k;
	return engine->LoadKernel(prec, valuesPerThread, &k);
}


////////////////////////////////////////////////////////////////////////////////
// Create and destroy sparse matrices

// sparse value, column, and row data are deinterleaved.
sparseStatus_t SPARSEAPI sparseMatCreate(sparseEngine_t engine, int height, int width,
	sparsePrec_t prec, int valuesPerThread, sparseInput_t input, int numElements,
	const void* sparse, const int* row, const int* col, sparseMat_t* matrix_) {

	ssContext c(engine->context);

	if(SPARSE_INPUT_CSR != input && SPARSE_INPUT_COO != input) 
		return SPARSE_STATUS_INVALID_VALUE;

	sparseStatus_t status = SPARSE_STATUS_INTERNAL_ERROR;
	std::auto_ptr<sparseMatrix> matrixData;

	try {
		switch(prec) {		
			case SPARSE_PREC_REAL4:
			case SPARSE_PREC_REAL_MIXED: {
				std::auto_ptr<EncodedMatrixData<float> > data;
				EncodeMatrixDeinterleaved(height, width, 32, valuesPerThread, input, 
					numElements, static_cast<const float*>(sparse), col, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}

			case SPARSE_PREC_REAL8: {
				std::auto_ptr<EncodedMatrixData<double> > data;
				EncodeMatrixDeinterleaved(height, width, 32, valuesPerThread, input,
					numElements, static_cast<const double*>(sparse), col, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX4:
			case SPARSE_PREC_COMPLEX_MIXED: {
				std::auto_ptr<EncodedMatrixData<cfloat> > data;
				EncodeMatrixDeinterleaved(height, width, 32, valuesPerThread, input,
					numElements, static_cast<const cfloat*>(sparse), col, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX8: {
				std::auto_ptr<EncodedMatrixData<cdouble> > data;
				EncodeMatrixDeinterleaved(height, width, 32, valuesPerThread, input, 
					numElements, static_cast<const cdouble*>(sparse), col, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			default:
				return SPARSE_STATUS_INVALID_VALUE;
		}
	} catch(std::bad_alloc) {
		status = SPARSE_STATUS_HOST_ALLOC_FAILED;
	} catch(RowOutOfOrder) {
		status = SPARSE_STATUS_NOT_SORTED;
	}

	if(SPARSE_STATUS_SUCCESS == status)
		*matrix_ = matrixData.release();

	return status;
}

// sparse value, column, and row data are interleaved. For CSR, a row array
// is still passed in. For COO, pass null for row.
sparseStatus_t SPARSEAPI sparseMatCreateInterleave(sparseEngine_t engine, int height,
	int width, sparsePrec_t prec, int valuesPerThread, sparseInput_t input, 
	int numElements, const void* sparse, const int* row, sparseMat_t* matrix_) {

	ssContext c(engine->context);

	// Test that the kernel exists
	sparseEngine_d::Kernel* k;
	sparseStatus_t status = engine->LoadKernel(prec, valuesPerThread, &k);
	if(SPARSE_STATUS_SUCCESS != status) return status;

	status = SPARSE_STATUS_INTERNAL_ERROR;
	std::auto_ptr<sparseMatrix> matrixData;

	try {
		switch(prec) {		
			case SPARSE_PREC_REAL4:
			case SPARSE_PREC_REAL_MIXED: {
				std::auto_ptr<EncodedMatrixData<float> > data;
				EncodeMatrixInterleaved(height, width, WARP_SIZE, valuesPerThread, input,
					numElements, sparse, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}

			case SPARSE_PREC_REAL8: {
				std::auto_ptr<EncodedMatrixData<double> > data;
				EncodeMatrixInterleaved(height, width, WARP_SIZE, valuesPerThread, input, 
					numElements, sparse, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX4:
			case SPARSE_PREC_COMPLEX_MIXED: {
				std::auto_ptr<EncodedMatrixData<cfloat> > data;
				EncodeMatrixInterleaved(height, width, WARP_SIZE, valuesPerThread, input, 
					numElements, sparse, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX8: {
				std::auto_ptr<EncodedMatrixData<cdouble> > data;
				EncodeMatrixInterleaved(height, width, WARP_SIZE, valuesPerThread, input,
					numElements, sparse, row, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			default:
				return SPARSE_STATUS_INVALID_VALUE;
		}
	} catch(std::bad_alloc) {
		status = SPARSE_STATUS_HOST_ALLOC_FAILED;
	} catch(RowOutOfOrder) {
		status = SPARSE_STATUS_NOT_SORTED;
	}
	
	if(SPARSE_STATUS_SUCCESS == status)
		*matrix_ = matrixData.release();

	return status;
}

// Pass in the appropriately typed callback function. This should simply
// copy data into the provided pointers, up to the amount requested. Returning
// fewer elements than requested indicates the end of the stream.
sparseStatus_t SPARSEAPI sparseMatCreateStream(sparseEngine_t engine, int height, 
	int width, sparsePrec_t prec, int valuesPerThread, int sizeHint, void* streamFp,
	void* cookie, sparseMat_t* matrix_) {

	ssContext c(engine->context);

	sparseStatus_t status = SPARSE_STATUS_INTERNAL_ERROR;
	std::auto_ptr<sparseMatrix> matrixData;

	try {
		switch(prec) {
			case SPARSE_PREC_REAL4:
			case SPARSE_PREC_REAL_MIXED: {
				std::auto_ptr<EncodedMatrixData<float> > data;
				EncodeMatrixStream(height, width, WARP_SIZE, valuesPerThread, sizeHint,
					(sparseStreamReal4_fp)streamFp, cookie, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}

			case SPARSE_PREC_REAL8: {
				std::auto_ptr<EncodedMatrixData<double> > data;
				EncodeMatrixStream(height, width, WARP_SIZE, valuesPerThread, sizeHint,
					(sparseStreamReal8_fp)streamFp, cookie, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX4:
			case SPARSE_PREC_COMPLEX_MIXED: {
				std::auto_ptr<EncodedMatrixData<cfloat> > data;
				EncodeMatrixStream(height, width, WARP_SIZE, valuesPerThread, sizeHint, 
					(sparseStreamComplex4_fp)streamFp, cookie, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			case SPARSE_PREC_COMPLEX8: {
				std::auto_ptr<EncodedMatrixData<cdouble> > data;
				EncodeMatrixStream(height, width, WARP_SIZE, valuesPerThread, sizeHint, 
					(sparseStreamComplex8_fp)streamFp, cookie, &data);
				status = CreateSparseMatrix(engine, *data, prec, &matrixData);
				break;
			}
			default:
				return SPARSE_STATUS_INVALID_VALUE;
		}
	} catch(std::bad_alloc) {
		status = SPARSE_STATUS_HOST_ALLOC_FAILED;
	} catch(RowOutOfOrder) {
		status = SPARSE_STATUS_NOT_SORTED;
	}

	if(SPARSE_STATUS_SUCCESS == status)
		*matrix_ = matrixData.release();

	return status;
}

sparseStatus_t SPARSEAPI sparseMatCreateGPU(sparseEngine_t engine,
	int height, int width, sparsePrec_t prec, int valuesPerThread, 
	int numElements, CUdeviceptr row, CUdeviceptr col, CUdeviceptr val, 
	sparseMat_t* matrix) {

	sparseEngine_d* e = static_cast<sparseEngine_d*>(engine);
	std::auto_ptr<sparseMatrix> sparse;
	sparseStatus_t status = e->Encode(height, width, prec, valuesPerThread, 
		numElements, row, col, val, &sparse);
	if(SPARSE_STATUS_SUCCESS == status)
		*matrix = sparse.release();

	return status;
}

sparseStatus_t SPARSEAPI sparseMatDestroy(sparseMat_t matrix_) {
	sparseMatrix* matrix = static_cast<sparseMatrix*>(matrix_);
	ssContext c(matrix->engine->context);

	delete matrix;

	return SPARSE_STATUS_SUCCESS;
}

sparseStatus_t SPARSEAPI sparseMatEngine(sparseMat_t matrix_, sparseEngine_t* engine) {
	sparseMatrix* matrix = static_cast<sparseMatrix*>(matrix_);
	matrix->engine->AddRef();
	*engine = matrix->engine;
	return SPARSE_STATUS_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix * vector multiplication exports


sparseStatus_t SPARSEAPI sparseMatVecSMul(sparseEngine_t engine, float alpha, 
	sparseMat_t matrix, CUdeviceptr x, float beta, CUdeviceptr y, CUstream stream) {

	if(SPARSE_PREC_REAL4 != matrix->prec) return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y, stream);
}

sparseStatus_t SPARSEAPI sparseMatVecDMul(sparseEngine_t engine, double alpha, 
	sparseMat_t matrix, CUdeviceptr x, double beta, CUdeviceptr y, CUstream stream) {

	if(SPARSE_PREC_REAL8 != matrix->prec && SPARSE_PREC_REAL_MIXED != matrix->prec) 
		return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y, stream);
}

sparseStatus_t SPARSEAPI sparseMatVecCMul(sparseEngine_t engine, sparseComplex4_t alpha,
	sparseMat_t matrix, CUdeviceptr x, sparseComplex4_t beta, CUdeviceptr y, CUstream stream) {

	if(SPARSE_PREC_COMPLEX4 != matrix->prec) return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y, stream);
}

sparseStatus_t SPARSEAPI sparseMatVecZMul(sparseEngine_t engine, sparseComplex8_t alpha,
	sparseMat_t matrix, CUdeviceptr x, sparseComplex8_t beta, CUdeviceptr y, CUstream stream) {

	if(SPARSE_PREC_COMPLEX8 != matrix->prec && SPARSE_PREC_COMPLEX_MIXED != matrix->prec)
		return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y, stream);
}
