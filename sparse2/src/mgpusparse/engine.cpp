
#include "engine.h"
#include <sstream>

const char* PrecNames[4] = {
	"float",
	"double",
	"cfloat",
	"cdouble"
};
const PrecTerm PrecTerms[4] = {
	{ 4, CU_AD_FORMAT_FLOAT, 1, },
	{ 8, CU_AD_FORMAT_UNSIGNED_INT32, 2 },
	{ 8, CU_AD_FORMAT_FLOAT, 2 },
	{ 16, CU_AD_FORMAT_UNSIGNED_INT32, 4 }
};

int IndexFromVT(int vt) {
	for(int i(0); i < NumVT; ++i)
		if(ValuesPerThread[i] == vt) return i;
	return -1;
}

/*
void Zero(float& x) { x = 0; }
void Zero(double& x) { x = 0; }
void Zero(float2& x) { x = make_float2(0, 0); }
void Zero(double2& x) { x = make_double2(0, 0); }
*/

bool IsZero(float x) { return !x; }
bool IsZero(double x) { return !x; }
bool IsZero(float2 x) { return !x.x && x.y; }
bool IsZero(double2 x) { return !x.x && x.y; }

sparseStatus_t CreateSparseEngine(const char* kernelPath, EnginePtr* ppEngine) {

	EnginePtr engine(new sparseEngine_d);
	engine->kernelPath = kernelPath;

	AttachCuContext(&engine->context);
	engine->numSMs = engine->context->Device()->NumSMs();

	if(2 != engine->context->Device()->ComputeCapability().first)
		return SPARSE_STATUS_CONFIG_NOT_SUPPORTED;
	
	ppEngine->swap(engine);
	return SPARSE_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::LoadKernel

sparseStatus_t sparseEngine_d::LoadKernel(sparsePrec_t prec,
	sparseEngine_d::Kernel** ppKernel) {
	
	// First attempt to load the finalize module if it is not yet loaded.
	CUresult result = CUDA_SUCCESS;

	// Check if the requested kernel is available, and if not, load it.
	int p = (int)prec;
	if(!multiply[p].get()) {
		std::auto_ptr<Kernel> k(new Kernel);
		
		std::string filename = kernelPath + "spmxv_" + PrecNames[p] +
			".cubin";
		result = context->LoadModuleFilename(filename, &k->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;

		// Load the five SpMxV kernels for different valuesPerThread counts.
		for(int i(0); i < NumVT; ++i) {
			std::ostringstream oss;
			oss<< "SpMxV_"<< ValuesPerThread[i];
			result = k->module->GetFunction(oss.str(), 
				make_int3(BlockSize, 1,1), &k->func[i]);
			if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
		}

		// Load the finalize function.
		result = k->module->GetFunction("Finalize", make_int3(BlockSize, 1, 1), 
			&k->finalize);
			if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		// Cache the texture reference
		result = cuModuleGetTexRef(&k->xVec_texture, k->module->Handle(),
			"xVec_texture");
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		result = cuTexRefSetFlags(k->xVec_texture, CU_TRSF_READ_AS_INTEGER);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		result = cuTexRefSetFormat(k->xVec_texture, PrecTerms[p].vecFormat, 
			PrecTerms[p].vecChannels);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		multiply[p] = k;
	}

	*ppKernel = multiply[p].get();
	return SPARSE_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::Multiply

template<typename T>
sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, T alpha, T beta,
	CUdeviceptr xVec, CUdeviceptr yVec) {

	sparseMatrix* m = static_cast<sparseMatrix*>(mat);

	Kernel* k;
	sparseStatus_t status = LoadKernel(m->prec, &k);
	if(SPARSE_STATUS_SUCCESS != status) return status;

	// Push the args and select the xVec as a texture
	CuCallStack callStack;
	callStack.Push(m->outputIndices, m->colIndices, m->sparseValues,
		m->tempOutput, m->numGroups);

	// Get the size of the xVec elements
	PrecTerm precTerms = PrecTerms[m->prec];	
	size_t offset;
	CUresult result = cuTexRefSetAddress(&offset, k->xVec_texture, xVec, 
		m->width * precTerms.vecSize);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
	
	// Launch the function
	uint numBlocks = DivUp(m->numGroups, WarpsPerBlock);
	result = k->func[IndexFromVT(m->valuesPerThread)]->Launch(numBlocks, 1, 
		callStack);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;

	// Finalize the vector
	int numFinalizeBlocks = DivUp(m->numGroups, WarpsPerBlock);
	int useBeta = !IsZero(beta);

	callStack.Reset();
	callStack.Push(m->tempOutput, m->rowIndices, m->height, yVec, alpha, beta,
		useBeta);

	result = k->finalize->Launch(numFinalizeBlocks, 1, callStack);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

	return SPARSE_STATUS_SUCCESS;
}


template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, float alpha,
	float beta, CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, double alpha,
	double beta, CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, float2 alpha,
	float2 beta, CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, double2 alpha,
	double2 beta, CUdeviceptr xVec, CUdeviceptr yVec);

