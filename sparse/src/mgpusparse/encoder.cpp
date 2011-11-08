
#include "engine.h"
#include <cassert>

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;

const int STORE_FLAG = 1<< 25;


void CSRToCOO(const int* csr, int height, int nz, std::vector<int>& coo) {
	coo.resize(nz);
	for(int i(0); i < height; ++i) {
		int a = csr[i];
		int b = csr[i + 1];
		std::fill(&coo[0] + a, &coo[0] + b, i);
	}
}


////////////////////////////////////////////////////////////////////////////////
// EncodeMatrixCOO

template<typename T>
void EncodeMatrixCOO(int height, int width, int vt, int nz, 
	const int* rowIndicesIn, const int* colIndicesIn, const T* valuesIn,
	std::auto_ptr<EncodedMatrix<T> >* ppMatrix) {

	std::auto_ptr<EncodedMatrix<T> > m(new EncodedMatrix<T>);
	m->height = height;
	m->width = width;
	m->valuesPerThread = vt;
	m->nz = nz;
	m->colIndices.reserve((int)(1.02 * nz));
	m->sparseValues.reserve((int)(1.02 * nz));
	m->rowIndices.resize(height + 1);

	int count = WarpSize * vt;
	std::vector<int> rows(count);
	std::vector<uint> cols(count);
	std::vector<T> vals(count);
	
	int prevRow = -1;

	int curOut = 0;

	int outputIndex = 0;

	int lastStoredRow = 0;
	int lastStoredCount = 0;

	while(nz) {

		////////////////////////////////////////////////////////////////////////
		// Read in the rows, cols, and values used in this block.

		int end = std::min(nz, count);
		int firstRow = *rowIndicesIn;

		for(int i(0); i < count; ++i) {
				
			int row = 0x7fffffff;
			int col = 0;
			T val = 0;

			if(i < end) {
				row = rowIndicesIn[i];
				if(row >= firstRow + WarpSize) 
					end = i;
			}

			if(i < end) {
				col = colIndicesIn[i];
				val = valuesIn[i];
				prevRow = row;
			} else
				row = prevRow;
			
			rows[i] = row;
			cols[i] = col;
			vals[i] = val;
		}


		////////////////////////////////////////////////////////////////////////
		// Compute the head flags.
		// laneRows holds the row index for each shared mem slot.
		// laneStarts holds the first special index.

		int sharedRows[64];
		int laneStarts[32];
		int storeSlots = 0;

		for(int lane(0); lane < WarpSize; ++lane) {
			int* r = &rows[vt * lane];
			uint* c = &cols[vt * lane];
			laneStarts[lane] = storeSlots;
			
			int laneCount = 0;
			for(int j(0); j < vt; ++j) {
				int curRow = r[j];
				if(j < vt - 1) {
					int nextRow = r[j + 1];
					if(curRow < nextRow) {
						c[j] |= STORE_FLAG;
						sharedRows[storeSlots + laneCount++] = curRow;
					}
				} else {
					// Don't need to actually set the store flag on the last 
					// value in each eval thread, as it is implied.
					//	c[j] |= STORE_FLAG;
					sharedRows[storeSlots + laneCount++] = curRow;
				}
			}
			storeSlots += laneCount;
		}

		////////////////////////////////////////////////////////////////////////
		// Compute the scan offsets. 

		int rowStarts[32], rowLast[32];
		std::fill(rowStarts, rowStarts + WarpSize, 0x7fffffff);
		std::fill(rowLast, rowLast + WarpSize, -1);

		// Find the first and last indices for each row within the shared
		// array.
		for(int i(0); i < storeSlots; ++i) {
			int row = sharedRows[i] - firstRow;
			rowStarts[row] = std::min(rowStarts[row], i);
			rowLast[row] = std::max(rowLast[row], i);
		}

		// Compute the distanceX and distanceY terms for each thread.
		int distances[64];
		for(int i(0); i < 64; ++i) {
			if(i < storeSlots) {
				int row = sharedRows[i] - firstRow;
				distances[i] = i - rowStarts[row];
			} else
				distances[i] = 0;
		}

		// Find the unique set of rows encountered in this group.
		int* sharedEnd = std::unique(sharedRows, sharedRows + storeSlots);
		int encountered = sharedEnd - sharedRows;

		int rowSumOffsets[32];
		for(int i(0); i < 32; ++i) {
			if(i < encountered) {
				int row = sharedRows[i] - firstRow;
				rowSumOffsets[i] = rowLast[row];
			} else
				rowSumOffsets[i] = 0;
		}


		////////////////////////////////////////////////////////////////////////
		// Bake the offsets into the column indices.

		for(int lane(0); lane < 32; ++lane) {
			int index = lane * vt;
			cols[index + 0] |= laneStarts[lane]<< 26;
			cols[index + 1] |= ((lane < encountered)<< 26) | 
				(distances[lane]<< 27);
			cols[index + 2] |= distances[32 + lane]<< 26;
			cols[index + 3] |= rowSumOffsets[lane]<< 26;
		}

		
		////////////////////////////////////////////////////////////////////////
		// Transpose the column indices and sparse values into 
		// EncodedMatrix.

		m->colIndices.resize(curOut + count);
		m->sparseValues.resize(curOut + count);
		uint* c = &m->colIndices[curOut];
		T* v = &m->sparseValues[curOut];

		for(int lane(0); lane < WarpSize; ++lane)
			for(int j(0); j < vt; ++j) {
				int source = vt * lane + j;
				int dest = j * WarpSize + lane;
				c[dest] = cols[source];
				v[dest] = vals[source];

			}


		////////////////////////////////////////////////////////////////////////
		// Prepare the output indices and row indices.

		m->outputIndices.push_back(outputIndex);

		for(int i(0); i < encountered; ++i) {
			int row = sharedRows[i];
			
			for(int j(lastStoredRow + 1); j <= row; ++j)
				m->rowIndices[j] = outputIndex;
		
			++outputIndex;
			lastStoredRow = row;
		}

		rowIndicesIn += end;
		colIndicesIn += end;
		valuesIn += end;
		nz -= end;
		curOut += count;
	}

	m->rowIndices[height] = outputIndex;
	m->nz2 = (int)m->colIndices.size();
	m->numGroups = (int)m->outputIndices.size();
	m->outputSize = outputIndex;
	m->packedSizeShift = 0;
	
	*ppMatrix = m;
}


////////////////////////////////////////////////////////////////////////////////
// CreateSparseMatrix

template<typename T>
sparseStatus_t CreateSparseMatrix(sparseEngine_t engine, 
	const EncodedMatrix<T>& data, sparsePrec_t prec, 
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
		data.outputSize * PrecTerms[prec].vecSize, &m->tempOutput);
	if(CUDA_SUCCESS != result) return SPARSE_STATUS_DEVICE_ALLOC_FAILED;

	m->storage = m->sparseValues->Size() + m->colIndices->Size() +
		m->rowIndices->Size() + m->outputIndices->Size() +
		m->tempOutput->Size();

	*ppMatrix = m;

	return SPARSE_STATUS_SUCCESS;
}


sparseStatus_t SPARSEAPI sparseMatCreate(sparseEngine_t engine, int height,
	int width, sparsePrec_t prec, int vt, sparseInput_t input, int nz,
	const void* sparse, const int* row, const int* col, sparseMat_t* matrix) {

	// Expand CSR to COO
	std::vector<int> cooStorage;
	const int* coo = row;
	if(SPARSE_INPUT_CSR == input) {
		CSRToCOO(row, height, nz, cooStorage);
		coo = &cooStorage[0];
	}		

	std::auto_ptr<sparseMatrix> m2;
	sparseStatus_t status;
	switch(prec) {
		case SPARSE_PREC_REAL4: {
			std::auto_ptr<EncodedMatrix<float> > m(new EncodedMatrix<float>);
			EncodeMatrixCOO<float>(height, width, vt, nz, coo, col,
				(const float*)sparse, &m);
			status = CreateSparseMatrix(engine, *m, prec, &m2);
			break;
		}
		case SPARSE_PREC_REAL8: {
			std::auto_ptr<EncodedMatrix<double> > m(new EncodedMatrix<double>);
			EncodeMatrixCOO<double>(height, width, vt, nz, coo, col,
				(const double*)sparse, &m);
			status = CreateSparseMatrix(engine, *m, prec, &m2);
			break;
		}
		case SPARSE_PREC_COMPLEX4: {
			std::auto_ptr<EncodedMatrix<cfloat> > m(new EncodedMatrix<cfloat>);
			EncodeMatrixCOO<cfloat>(height, width, vt, nz, coo, col,
				(const cfloat*)sparse, &m);
			status = CreateSparseMatrix(engine, *m, prec, &m2);
			break;
		}
		case SPARSE_PREC_COMPLEX8: {
			std::auto_ptr<EncodedMatrix<cdouble> >
				m(new EncodedMatrix<cdouble>);
			EncodeMatrixCOO<cdouble>(height, width, vt, nz, coo, col,
				(const cdouble*)sparse, &m);
			status = CreateSparseMatrix(engine, *m, prec, &m2);
			break;
		}
	}

	if(SPARSE_STATUS_SUCCESS == status)
		*matrix = m2.release();
	return status;
}
