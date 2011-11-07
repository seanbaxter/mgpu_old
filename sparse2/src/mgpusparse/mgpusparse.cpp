
#include "engine.h"
#include <vector>



const char* SparseStatusStrings[] = {
	"SPARSE_STATUS_SUCCESS",
	"SPARSE_STATUS_NOT_INITIALIZED",
	"SPARSE_STATUS_DEVICE_ALLOC_FAILED",
	"SPARSE_STATUS_HOST_ALLOC_FAILED",
	"SPARSE_STATUS_PREC_MISMATCH",
	"SPARSE_STATUS_CONFIG_NOT_SUPPORTED",
	"SPARSE_STATUS_CONTEXT_MISMATCH",
	"SPARSE_STATUS_INVALID_CONTEXT",
	"SPARSE_STATUS_NOT_SORTED",
	"SPARSE_STATUS_KERNEL_NOT_FOUND",
	"SPARSE_STATUS_KERNEL_ERROR",
	"SPARSE_STATUS_LAUNCH_ERROR",
	"SPARSE_STATUS_INVALID_VALUE",
	"SPARSE_STATUS_DEVICE_ERROR",
	"SPARSE_STATUS_INTERNAL_ERROR"
};

const char* SPARSEAPI sparseStatusString(sparseStatus_t status) {
	int code = (int)status;
	if(code >= (int)(sizeof(SparseStatusStrings) / sizeof(char*))) return 0;
	return SparseStatusStrings[code];
}


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

sparseStatus_t SPARSEAPI sparseRelease(sparseEngine_t engine) {
	engine->Release();
	return SPARSE_STATUS_SUCCESS;
}

sparseStatus_t SPARSEAPI sparseQuerySupport(sparseEngine_t engine,
	sparsePrec_t prec) {

	sparseEngine_d::Kernel* k;
	return engine->LoadKernel(prec, &k);
}


////////////////////////////////////////////////////////////////////////////////
// Create and destroy sparse matrices


sparseStatus_t SPARSEAPI sparseMatDestroy(sparseMat_t matrix_) {

	sparseMatrix* matrix = static_cast<sparseMatrix*>(matrix_);
	delete matrix;

	return SPARSE_STATUS_SUCCESS;
}

sparseStatus_t SPARSEAPI sparseMatEngine(sparseMat_t matrix_, 
	sparseEngine_t* engine) {

	sparseMatrix* matrix = static_cast<sparseMatrix*>(matrix_);
	matrix->engine->AddRef();
	*engine = matrix->engine;
	return SPARSE_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// Matrix * vector multiplication exports

sparseStatus_t SPARSEAPI sparseMatVecSMul(sparseEngine_t engine, float alpha, 
	sparseMat_t matrix, CUdeviceptr x, float beta, CUdeviceptr y) {

	if(SPARSE_PREC_REAL4 != matrix->prec) return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y);
}

sparseStatus_t SPARSEAPI sparseMatVecDMul(sparseEngine_t engine, double alpha, 
	sparseMat_t matrix, CUdeviceptr x, double beta, CUdeviceptr y) {

	if(SPARSE_PREC_REAL8 != matrix->prec) 
		return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, alpha, beta, x, y);
}

sparseStatus_t SPARSEAPI sparseMatVecCMul(sparseEngine_t engine, 
	sparseComplex4_t alpha, sparseMat_t matrix, CUdeviceptr x, 
	sparseComplex4_t beta, CUdeviceptr y) {

	if(SPARSE_PREC_COMPLEX4 != matrix->prec) return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, make_float2(alpha.real(), alpha.imag()),
		make_float2(beta.real(), beta.imag()), x, y);
}

sparseStatus_t SPARSEAPI sparseMatVecZMul(sparseEngine_t engine, 
	sparseComplex8_t alpha, sparseMat_t matrix, CUdeviceptr x, 
	sparseComplex8_t beta, CUdeviceptr y) {

	if(SPARSE_PREC_COMPLEX8 != matrix->prec)
		return SPARSE_STATUS_PREC_MISMATCH;
	return engine->Multiply(matrix, make_double2(alpha.real(), alpha.imag()),
		make_double2(beta.real(), beta.imag()), x, y);
}
