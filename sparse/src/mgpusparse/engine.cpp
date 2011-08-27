
#include "engine.h"
#include <sstream>

const char* PrecisionNames[6] = {
	"float",
	"float_double",
	"double",
	"cfloat",
	"cfloat_double",
	"cdouble"
};

struct PrecTerm {
	int vecSize;
	CUarray_format vecFormat;
	int vecChannels;

	int tempSize;
	CUarray_format tempFormat;
	int tempChannels;	
};

const PrecTerm PrecTerms[6] = {
	{ 4, CU_AD_FORMAT_FLOAT, 1, 4, CU_AD_FORMAT_FLOAT, 1 },
	{ 4, CU_AD_FORMAT_FLOAT, 1, 8, CU_AD_FORMAT_UNSIGNED_INT32, 2 },
	{ 8, CU_AD_FORMAT_UNSIGNED_INT32, 2, 8, CU_AD_FORMAT_UNSIGNED_INT32, 2 },
	{ 8, CU_AD_FORMAT_FLOAT, 2, 8, CU_AD_FORMAT_FLOAT, 2 },
	{ 8, CU_AD_FORMAT_FLOAT, 2, 16, CU_AD_FORMAT_UNSIGNED_INT32, 4 },
	{ 16, CU_AD_FORMAT_UNSIGNED_INT32, 4, 16, CU_AD_FORMAT_UNSIGNED_INT32, 4 }
};

// TODO: compute block shape for sm_13
// This is sm_20
int ComputeOptimalBlockShape(int r) {

	r = ~1 & (r + 1);
	
	// how many threads can we support?
	int maxThreads = ~31 & std::min(32768 / r, 1536);

	// how many equal sized blocks can we break this up into?
	int optimumBlockCount = 1;
	int optimumBlockSize = std::min(maxThreads, 512);
	int optimumBlockTotalThreads = optimumBlockSize;

	// find the smallest blocks that give the best occupancy
	for(int blocks(2); blocks <= 8; ++blocks) {
		int threadsPerBlock = std::min(~31 & (maxThreads / blocks), 512);
		int blockTotalThreads = blocks * threadsPerBlock;
		if(blockTotalThreads >= optimumBlockTotalThreads) {
			optimumBlockCount = blocks;
			optimumBlockSize = threadsPerBlock;
			optimumBlockTotalThreads = blockTotalThreads;
		}
	}

	return optimumBlockSize;
}

sparseStatus_t CreateSparseEngine(const char* kernelPath, EnginePtr* ppEngine) {

	EnginePtr engine(new sparseEngine_d);
	engine->kernelPath = kernelPath;

	AttachCuContext(&engine->context);
	engine->numSMs = engine->context->Device()->NumSMs();
	
	ppEngine->swap(engine);
	return SPARSE_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::LoadKernel

sparseStatus_t sparseEngine_d::LoadKernel(sparsePrec_t prec,
	sparseEngine_d::Kernel** ppKernel) {
	

	// First attempt to load the finalize module if it is not yet loaded.
	CUresult result = CUDA_SUCCESS;
	if(!finalize.get()) {
		std::auto_ptr<Finalize> f(new Finalize);

		std::string filename = kernelPath + "finalize.cubin";
		result = context->LoadModuleFilename(filename, &f->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;
		
		// load the finalize functions from the module
		const char* FinalizeSuffix[4] = {
			"float", "double", "cfloat", "cdouble"
		};
		for(int i(0); i < 2; ++i)
			for(int j(0); j < 4; ++j) {
				std::ostringstream oss;
				oss<< "Finalize_"<< FinalizeSuffix[j];
				if(i) oss<< "_special";
				result = f->module->GetFunction(oss.str(), make_int3(128, 1, 1),
					&f->func[j][i]);
				if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
			}
		finalize = f;
	}


	// Check if the requested kernel is available, and if not, load it.
	int p = (int)prec;
	if(!multiply[p].get()) {
		std::auto_ptr<Kernel> k(new Kernel);
		
		std::string filename = kernelPath + "spmxv_" + PrecisionNames[p] +
			".cubin";
		result = context->LoadModuleFilename(filename, &k->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;

		for(int i(0); i < 5; ++i) {
			std::ostringstream oss;
			oss<< "SpMxV_"<< (4 * i + 4);
			result = k->module->GetFunction(oss.str(), 
				make_int3(NumThreads, 1,1), &k->func[i]);
			if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
		}

		// Cache the texture reference
		CUmodule module = k->module->Handle();
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


///////////////////////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::LoadBuild

bool GatherScatterPacked(int warpSize, int valuesPerThread, std::vector<uint>& pairs);

sparseStatus_t sparseEngine_d::LoadBuild(sparsePrec_t prec, int valuesPerThread,
	sparseEngine_d::Build** ppBuild) {
/*
	CUresult result = CUDA_SUCCESS;
	if(!count.get()) {
		std::auto_ptr<Count> c(new Count);
		std::string filename = kernelPath + "dencode_count.ptx";
		result = LoadModuleFilename(filename.c_str(), &c->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;
	
		result = cuModuleGetFunction(&c->func, c->module->hmod, "DeviceEncode_CountBlocks");
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		c->valuesPerThread = 0;
		count = c;
	}
	if(!rebuild.get()) {
		std::auto_ptr<RebuildIndices> r(new RebuildIndices);
		std::string filename = buildPath + "rebuild_rowindices.ptx";
		result = LoadModuleFilename(filename.c_str(), &r->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;
	
		result = cuModuleGetFunction(&r->func, r->module->hmod, "RebuildRowIndices");
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		result = cuFuncSetBlockShape(r->func, 256, 1, 1);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		rebuild = r;
	}

	int valSlot = valuesPerThread / 4 - 1;
	int vecSize = PrecTerms[(int)prec].vecSize;
	int sizeSlot;
	switch(vecSize) {
		case 4: sizeSlot = 0; break;
		case 8: sizeSlot = 1; break;
		case 16: sizeSlot = 2; break;
	}
	if(!build[sizeSlot][valSlot].get()) {
		std::auto_ptr<Build> b(new Build);
		std::ostringstream oss;
		oss<< buildPath<< "dencode_build_"<< valuesPerThread<< "_"<< vecSize<< ".ptx";
		result = LoadModuleFilename(oss.str().c_str(), &b->module);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_NOT_FOUND;

		result = cuModuleGetFunction(&b->func, b->module->hmod, "DeviceEncode_BuildBlocks");
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		int numRegs, sharedSize;
		cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, b->func);
		cuFuncGetAttribute(&sharedSize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, b->func);
		
		int total = valuesPerThread * warpSize;
		int regsPerBlock = total * (~1 & (numRegs + 1));
		int blocksPerMP = std::min((48 * 1024) / sharedSize, 32768 / regsPerBlock);

		b->blocks = numMPs * std::min(8, blocksPerMP);

		result = cuFuncSetBlockShape(b->func, total, 1, 1);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;


			// NOTE:
			// for valuesPerThread that are powers of two, use arithmetic expression for
			// scatter/gather
			CUdeviceptr sharedGlobal;
			size_t size;
			result = cuModuleGetGlobal(&sharedGlobal, &size, b->module->hmod, "GatherScatter");
			if((CUDA_SUCCESS != result) || (size != 4 * total))
				return SPARSE_STATUS_KERNEL_ERROR;

			std::vector<uint> gatherScatter;
			GatherScatterPacked(warpSize, valuesPerThread, gatherScatter);
			result = cuMemcpyHtoD(sharedGlobal, &gatherScatter[0], size);
			if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;


		build[sizeSlot][valSlot] = b;
	}
	*ppBuild = build[sizeSlot][valSlot].get();
*/
	return SPARSE_STATUS_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::Multiply

template<typename T>
sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat_, T alpha, T beta,
	CUdeviceptr xVec, CUdeviceptr yVec) {

	sparseMatrix* m = static_cast<sparseMatrix*>(mat_);
	CuContext* c = m->engine->context.get();

	Kernel* k;
	sparseStatus_t status = LoadKernel(m->prec, &k);
	if(SPARSE_STATUS_SUCCESS != status) return status;

	// Push the args and select the xVec as a texture
	CuCallStack callStack;
	callStack.Push(m->rowIndices, m->colIndices, m->sparseValues, m->numGroups,
		m->tempOutput);


	// get the size of the xVec elements
	PrecTerm precTerms = PrecTerms[m->prec];	
	size_t offset;
	CUresult result = cuTexRefSetAddress(&offset, k->xVec_texture, xVec, 
		m->width * precTerms.vecSize);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
	
	// launch the function
	uint numBlocks = DivUp(m->numGroups, WarpsPerBlock);
	result = k->func[m->valuesPerThread / 4 - 1]->Launch(numBlocks, 1, 
		callStack);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;



	/*

	// Finalize the vector
	callStack.Reset();
	callStack.Push(mat->tempOutput->hmem, mat->outputIndices->hmem, mat->height, yVec);
	CUfunction finalizeFunc;
	if(T(1) == alpha && T(0) == beta) {
		if(mat->packedSizeShift) {
			// SpMxVFinalize1
			callStack.Push(mat->packedSizeShift);
			finalizeFunc = finalize[mat->prec]->func[0];
		} else
			// SpMxVFinalize2
			finalizeFunc = finalize[mat->prec]->func[1];		
	} else {
		if(mat->packedSizeShift) {
			// SpMxVFinalize3
			callStack.Push(mat->packedSizeShift, alpha, beta);
			finalizeFunc = finalize[mat->prec]->func[2];
		} else {
			// SpMxVFinalize4
			callStack.Push(alpha, beta);
			finalizeFunc = finalize[mat->prec]->func[3];
		}
	}

	result = callStack.SetToFunction(finalizeFunc);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

	numBlocks = (mat->height + 511) / 512;
	result = stream ? cuLaunchGridAsync(finalizeFunc, 16, (numBlocks + 15) / 16, stream) :
		cuLaunchGrid(finalizeFunc, 16, (numBlocks + 15) / 16);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;
*/
	return SPARSE_STATUS_SUCCESS;
}

template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, float alpha, float beta,
	CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, double alpha, double beta,
	CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, cfloat alpha, cfloat beta,
	CUdeviceptr xVec, CUdeviceptr yVec);
template sparseStatus_t sparseEngine_d::Multiply(sparseMat_t mat, cdouble alpha, cdouble beta,
	CUdeviceptr xVec, CUdeviceptr yVec);


///////////////////////////////////////////////////////////////////////////////////////////////////
// sparseEngine_d::Encode

sparseStatus_t sparseEngine_d::Encode(int height, int width, sparsePrec_t prec,
	int valuesPerThread, int nz, CUdeviceptr row, CUdeviceptr col, CUdeviceptr val, 
	std::auto_ptr<sparseMatrix>* ppMatrix) {

	Build* b;
	sparseStatus_t status = LoadBuild(prec, valuesPerThread, &b);
	if(SPARSE_STATUS_SUCCESS != status) return status;

	int total = WarpSize * valuesPerThread;
	int spacing = 2 * total;
	int numBlocks = std::min((nz + spacing - 1) / spacing, b->blocks);	

/*
	///////////////////////////////////////////////////////////////////////////////////////////////
	// run the count algorithm

	// allocate the shared memory and set the block shape for the count kernel
	CUresult result;
	if(valuesPerThread != count->valuesPerThread) {
		count->valuesPerThread = valuesPerThread;
		
		result = cuFuncSetBlockShape(count->func, total, 1, 1);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;

		result = cuFuncSetSharedSize(count->func, sizeof(uint) * 3 * total);
		if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
	}

	std::vector<int2> rangePairs(numBlocks);
	for(int i(0); i < numBlocks; ++i) {
		int target = (int)((i + 1) * (nz / (double)numBlocks));
		// round to the nearest multiple of total
		target -= target % total;		

		rangePairs[i] = make_int2(
			i ? rangePairs[i - 1].y : 0,
			(i == numBlocks - 1) ? nz : target);
	}
	ssDevicememPtr rangePairsDevice;
	result = AllocDeviceMem(rangePairs, &rangePairsDevice);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	ssDevicememPtr groupInfoDevice;
	result = AllocDeviceMem(sizeof(uint4) * numBlocks, 0, &groupInfoDevice);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	ssCallStack callStack;
	callStack.Push(total, row, rangePairsDevice->hmem, groupInfoDevice->hmem);

	result = callStack.SetToFunction(count->func);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_KERNEL_ERROR;
	
	result = cuLaunchGrid(count->func, numBlocks, 1);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;

	std::vector<int4> groupInfo;
	groupInfoDevice->ToHost(groupInfo);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// scan groupInfo and allocate resources for the encoded sparse matrix 

	// Make an exclusive sum of the group counts 
	std::vector<int4> groupInfo2(numBlocks);
	int totalGroups = 0;
	int totalTempOutput = 0;
	
	for(int i = 0; i < numBlocks; ++i) {
		// count the groups and map to STREAM_OUTPUT in .x
		groupInfo2[i].x = totalGroups;
		totalGroups += groupInfo[i].x;

		// count the temp output slots 
		groupInfo2[i].y = totalTempOutput;
		totalTempOutput += groupInfo[i].y;

		// the last row encountered by the preceding block is the start iterator for the
		// next block.
		groupInfo2[i].z = i ? (groupInfo[i - 1].z + 1) : 0;
	}

	std::auto_ptr<sparseMatrix> matrix(new sparseMatrix);
	
	matrix->height = height;
	matrix->width = width;
	matrix->prec = prec;

	matrix->warpSize = warpSize;
	matrix->valuesPerThread = valuesPerThread;
	matrix->numGroups = totalGroups;
	matrix->packedSizeShift = 0;	// TODO: UPDATE ME

	matrix->outputSize = totalTempOutput;
	matrix->nz = nz;
	matrix->nz2 = total * totalGroups;
	matrix->engine = this;

	int valueSize = PrecTerms[(int)prec].vecSize;
	result = AllocDeviceMem(sizeof(uint) * total * totalGroups, 0, &matrix->colIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = AllocDeviceMem(valueSize * total * totalGroups, 0, &matrix->sparseValues);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = AllocDeviceMem((height + 1) * sizeof(uint), 0, &matrix->outputIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = AllocDeviceMem(valueSize * totalTempOutput, 0, &matrix->tempOutput);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = AllocDeviceMem(sizeof(uint) * totalGroups, 0, &matrix->rowIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;


	///////////////////////////////////////////////////////////////////////////////////////////////
	// run the device encoder build kernel

	groupInfoDevice->FromHost(groupInfo2);

	callStack.Reset();
	callStack.Push(row, col, val, rangePairsDevice->hmem, groupInfoDevice->hmem, 
		matrix->colIndices->hmem, matrix->sparseValues->hmem, 
		matrix->rowIndices->hmem, matrix->outputIndices->hmem, height);

	result = callStack.SetToFunction(b->func);
	result = cuLaunchGrid(b->func, numBlocks, 1);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;


	///////////////////////////////////////////////////////////////////////////////////////////////
	// Rebuild the row indices to encode the element count in the high bits
	// of rowIndices.

	int maxBit = FindMaxBit(matrix->outputSize) + 1;
	
	ssDevicememPtr rebuiltIndices;
	result = AllocDeviceMem(matrix->width * 4, 0, &rebuiltIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	// NOTE: sm_20 only! This assumes 1536 threads per MP. It will run correctly
	// on other profiles but not necessarily optimally.
	int numRebuildMPs = 6 * numMPs;
	std::vector<int2> rangePairsRebuild(numRebuildMPs);
	for(int i(0); i < numRebuildMPs; ++i) {
		int end = (int)((i + 1) * (matrix->width / (double)numRebuildMPs));
		if(i) rangePairsRebuild[i].x = rangePairsRebuild[i - 1].y;
		rangePairsRebuild[i].y = (i == numRebuildMPs - 1) ? matrix->width : end;		
	}
	ssDevicememPtr rangePairsRebuildDevice;
	result = AllocDeviceMem(rangePairsRebuild, &rangePairsRebuildDevice);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	callStack.Reset();
	callStack.Push(matrix->outputIndices->hmem, rebuiltIndices->hmem,
		rangePairsRebuildDevice->hmem, maxBit);
	callStack.SetToFunction(rebuild->func);
	cuLaunchGrid(rebuild->func, numRebuildMPs, 1);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_LAUNCH_ERROR;

	rangePairsRebuildDevice->ToHost(rangePairsRebuild);
	int maxCount = 0;
	for(int i(0); i < numRebuildMPs; ++i)
		maxCount = std::max(maxCount, rangePairsRebuild[i].x);

	// Does maxCount fit into the allocated bits?
	int maxBit2 = FindMaxBit(maxCount) + 1;

	if(32 - maxBit2 > maxBit) {
		// Use the rebuild range pairs
		matrix->packedSizeShift = maxBit;
		matrix->outputIndices = rebuiltIndices;
	}

	matrix->storage = 
		matrix->sparseValues->size + 
		matrix->colIndices->size + 
		matrix->rowIndices->size +
		matrix->outputIndices->size +
		matrix->tempOutput->size;

	*ppMatrix = matrix;
*/
	return SPARSE_STATUS_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// CreateSparseMatrix

template<typename T>
sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrixData<T>& data, sparsePrec_t prec, 
	std::auto_ptr<sparseMatrix>* ppMatrix) {

	std::auto_ptr<sparseMatrix> m(new sparseMatrix);
	m->height = data.height;
	m->width = data.width;
	m->prec = prec;
	m->valuesPerThread = data.valuesPerThread;
	m->numGroups = data.numGroups;
	m->packedSizeShift = data.packedSizeShift;
	m->outputSize = data.outputSize;
	m->nz = data.nz;
	m->nz2 = (int)data.sparseValues.size();
	m->engine = engine;

	CUresult result = engine->context->MemAlloc(data.sparseValues, 
		&m->sparseValues);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = engine->context->MemAlloc(data.colIndices, &m->colIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = engine->context->MemAlloc(data.rowIndices, &m->rowIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = engine->context->MemAlloc(data.outputIndices, &m->outputIndices);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	result = engine->context->ByteAlloc(
		data.outputSize * PrecTerms[prec].tempSize, 0, &m->tempOutput);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	m->storage = m->sparseValues->Size() + m->colIndices->Size() +
		m->rowIndices->Size() + m->outputIndices->Size() +
		m->tempOutput->Size();

	*ppMatrix = m;

	return SPARSE_STATUS_SUCCESS;
}

template sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrixData<float>& data, sparsePrec_t prec, 
	std::auto_ptr<sparseMatrix>* ppMatrix);
template sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrixData<double>& data, sparsePrec_t prec, 
	std::auto_ptr<sparseMatrix>* ppMatrix);
template sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrixData<cfloat>& data, sparsePrec_t prec, 
	std::auto_ptr<sparseMatrix>* ppMatrix);
template sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrixData<cdouble>& data, sparsePrec_t prec, 
	std::auto_ptr<sparseMatrix>* ppMatrix);

