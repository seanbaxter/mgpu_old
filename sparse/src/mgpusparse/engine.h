#pragma once

#include "../../../inc/mgpusparse.h"
#include "../../../util/cucpp.h"
#include <string>

const int WarpsPerBlock = 8;
const int BlockSize = WarpSize * WarpsPerBlock;

const int NumVT = 7;
const int ValuesPerThread[NumVT] = {
	4, 6, 8, 10, 12, 16, 20
};

int IndexFromVT(int vt);


struct RowOutOfOrder { };

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;


// Copy these directly into CUdeviceptrs.
template<typename T>
struct EncodedMatrix {
	int height, width, valuesPerThread, nz, nz2, numGroups, 
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

struct PrecTerm {
	int vecSize;
	CUarray_format vecFormat;
	int vecChannels;
};
extern const PrecTerm PrecTerms[4];

struct sparseMatrix;
struct sparseEngine_d : public CuBase {
	
	struct Kernel {
		ModulePtr module;

		// Multiple function defined over 7 valuesPerThread sizes 
		// (4, 6, 8, 10, 12, 16, 20).
		FunctionPtr func[NumVT];

		// Single finalize function that supports both standard and BLAS-style
		// reduction.
		FunctionPtr finalize;

		// Texture reference for x-vector.
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

	sparseStatus_t Encode(int height, int width, sparsePrec_t prec, int vt,
		int nz, CUdeviceptr row, CUdeviceptr col, CUdeviceptr val, 
		std::auto_ptr<sparseMatrix>* ppMatrix);

	ContextPtr context;
	int numSMs;
	
	// Multiply kernel defined over 4 precisions.
	std::auto_ptr<Kernel> multiply[4];

	// Build kernel defined over three element sizes (4, 8, 16) and 5 
	// valuesPerThread sizes.
//	std::auto_ptr<Build> build[3][5];


	// a single count kernel serves all encoders. Set block shape before calling.
//	std::auto_ptr<Count> count;

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
