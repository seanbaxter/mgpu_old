#pragma once

#include "../../../inc/mgpusparse.h"
#include "../../../util/cucpp.h"
#include <string>

const int NumThreads = 128;
const int WarpsPerBlock = NumThreads / WarpSize;

struct RowOutOfOrder { };

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;


// Copy these directly into CUdeviceptrs.
template<typename T>
struct EncodedMatrixData {
	int height, width, warpSize, valuesPerThread, nz, nz2, numGroups, 
		outputSize, packedSizeShift;
	std::vector<T> sparseValues;
	std::vector<uint> colIndices, rowIndices, outputIndices;
};

// templates input types
template<typename T>
struct CSRElement {
	T value;
	int col;
};
template<typename T>
struct COOElement {
	T value;
	int row, col;
};


struct sparseMatrix;
struct sparseEngine_d : public CuBase {
	
	// Finalize joins together partial dot products. There are four functions,
	// one for each precision type.
	struct Finalize {
		ModulePtr module;
		FunctionPtr func[2][4];
	}; 
	struct Kernel {
		ModulePtr module;

		// Multiple function defined over 5 valuesPerThread sizes 
		// (4, 8, 12, 16, 20).
		FunctionPtr func[5];
		CUtexref xVec_texture;
	};
	struct Build {
		ModulePtr module;
		FunctionPtr func;
		int blocks;
	};
	struct Count {
		ModulePtr module;
		FunctionPtr func;
		int valuesPerThread;
	};
	struct RebuildIndices {
		ModulePtr module;
		FunctionPtr func;
	};

	sparseStatus_t LoadKernel(sparsePrec_t prec, Kernel** ppKernel);
	sparseStatus_t LoadBuild(sparsePrec_t prec, int valuesPerThread, 
		Build** ppBuild);

	template<typename T>
	sparseStatus_t Multiply(sparseMat_t mat, T alpha, T beta, CUdeviceptr xVec, 
		CUdeviceptr yVec);

	sparseStatus_t Encode(int height, int width, sparsePrec_t prec, int valuesPerThread,
		int nz, CUdeviceptr row, CUdeviceptr col, CUdeviceptr val, 
		std::auto_ptr<sparseMatrix>* ppMatrix);

	ContextPtr context;
	int numSMs;

	// All finalize kernels are defined in the same module.
	std::auto_ptr<Finalize> finalize;
	
	// Multiply kernel defined over 6 precisions.
	std::auto_ptr<Kernel> multiply[6];

	// Build kernel defined over three element sizes (4, 8, 16) and 5 
	// valuesPerThread sizes.
	std::auto_ptr<Build> build[3][5];


	// a single count kernel serves all encoders. Set block shape before calling.
	std::auto_ptr<Count> count;

	// a single rebuild kernel packs row element counts into the high bits of
	// rowIndices.
	std::auto_ptr<RebuildIndices> rebuild;

	std::string kernelPath;
};
typedef intrusive_ptr2<sparseEngine_d> EnginePtr;

sparseStatus_t CreateSparseEngine(const char* kernelPath, EnginePtr* ppEngine);

struct sparseMatrix : sparseMat_d {
	DeviceMemPtr sparseValues;
	DeviceMemPtr colIndices;
	DeviceMemPtr rowIndices;
	DeviceMemPtr outputIndices;
	DeviceMemPtr tempOutput;

	EnginePtr engine;
};

template<typename T>
sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, const EncodedMatrixData<T>& data,
	sparsePrec_t prec, std::auto_ptr<sparseMatrix>* ppMatrix);


// Unify all four encoding precisions (real4, real8, complex4, complex8) behind these
// templates
template<typename T>
void EncodeMatrixDeinterleaved(int height, int width, int warpSize, int valuesPerThread,
	sparseInput_t input, int nz, const T* sparse, const int* col, const int* row,
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix);

template<typename T>
void EncodeMatrixInterleaved(int height, int width, int warpSize, int valuesPerThread, 
	sparseInput_t input, int nz, const void* sparse, const int* row,
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix);

template<typename T>
void EncodeMatrixStream(int height, int width, int warpSize, int valuesPerThread,
	int sizeHint, int(SPARSEAPI*fp)(int, T*, int*, int*, void*), void* cookie,
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix);

