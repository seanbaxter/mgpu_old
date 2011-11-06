
#include "engine.h"
#include <cassert>


/*
const uint FirstThreadRow = 1<< 23;
const uint LastThreadRow = 1<< 24;
const uint SerializeRow = 1<< 25;

struct StreamResult {
	int consumed;
	int outputCount;
};

struct EncoderBuilder {

	// Process groupSize elements from rowIndices2, colIndices2.
	// Return the number of output slots.
	int ProcessWarp();

	// ConsumeAndWrite takes an input fragment, prepares a valid groupSize run 
	// of data, and calls ProcessWarp.
	template<typename T>
	StreamResult ConsumeAndWrite(const T* sparseValues, const int* colIndices,
		const int* rowIndices, int nz, T* sparseValuesOut, uint* colIndicesOut);

	int precedingRow;
	int valuesPerThread, groupSize;
	int nzCount;

	// WarpColIndices, sharedMemSlots, scanDeltas, and outputSlots are
	// written by ProcessWarp

	// Sized to groupSize - holds column indices and first/last flags in 
	// row-major order
	std::vector<uint> warpColIndices;
	std::vector<int> includedRowIndices;

	// Sized to 2 * warpSize
	// sharedMemSlots indicates that sharedArray[i]
	//		manages a partial sum ending at element sharedMemSlots[i]
	// scanDeltas indicates that thread tid should add in all values between
	//		tid + scanDeltas and tid for the parallel segmented scan.
	std::vector<int> sharedMemSlots;
	std::vector<int> scanDeltas;
	std::vector<int> outputSlots;
	std::vector<int> transposeIndices;

	// written by ConsumeAndWrite
	std::vector<int> rowIndices2, colIndices2;

	void Init(int valuesPerThread_) {
		precedingRow = 0;
		valuesPerThread = valuesPerThread_;
		groupSize = WarpSize * valuesPerThread;
		nzCount = 0;
		warpColIndices.resize(groupSize);
		includedRowIndices.resize(WarpSize);
		sharedMemSlots.resize(2 * WarpSize);
		scanDeltas.resize(2 * WarpSize);
		outputSlots.resize(WarpSize);
		rowIndices2.resize(groupSize);
		colIndices2.resize(groupSize);

		transposeIndices.resize(groupSize);
		for(int warp(0), i(0); warp < WarpSize; ++warp)
			for(int val(0); val < valuesPerThread; ++val, ++i)
				// scatter from i to the transposed index
				transposeIndices[i] = (i % valuesPerThread) * WarpSize +
					(i / valuesPerThread);
	}
};

int EncoderBuilder::ProcessWarp() {

	// Loop through each value in row-major order and build the colIndex flags.
	int curSharedMemSlot = 0;
//	int lastFlagCount = 0;
	for(int tid(0), i(0); tid < WarpSize; ++tid) {
		int curThreadRow = -1;
	//	int lastFlagCount = 0;
		int threadSharedMemSlot = curSharedMemSlot;
		for(int val(0); val < valuesPerThread; ++val, ++i) {
			int row = rowIndices2[i];
			uint col = (uint)colIndices2[i];
			if(row != curThreadRow) {
				// This is the first value of this row in this thread.
				col |= FirstThreadRow;

				// Set the previous LastThreadRow flag.
				if(val) {
					warpColIndices[i - 1] |= LastThreadRow;
					sharedMemSlots[curSharedMemSlot++] = i - 1;
				}

				if(row < precedingRow) throw RowOutOfOrder();
				precedingRow = row;
							
				curThreadRow = row;
			}
			warpColIndices[i] = col;
		}
		// Set the previous LastThreadRow flagg.
		warpColIndices[i - 1] |= LastThreadRow;
		sharedMemSlots[curSharedMemSlot++] = i - 1;

		// Write the first shared mem slot to the first colIndex for this 
		// thread.
		warpColIndices[i - valuesPerThread] |= threadSharedMemSlot<< 25;
	}

	// We perform a parallel segmented scan over 2 * warpSize elements.
	// Each thread updates slots tid and tid + warpSize.
	// Find the first shared mem slot that has the same row index as the 
	// current slot in the scan and store the difference as deltaX and deltaY.
//	int curScanRow = -1;
	int precedingSlotRow = -1;
	int precedingSlotStart = -1;
	int numOutputSlots = 0;
	for(int tid(0); tid < 2 * WarpSize; ++tid) {
		uint delta = 0;
		if(tid < curSharedMemSlot) {
			int sharedSlotRow = rowIndices2[sharedMemSlots[tid]];
			if(sharedSlotRow != precedingSlotRow) {
				if(tid) {
					outputSlots[numOutputSlots] = tid - 1;
					includedRowIndices[numOutputSlots++] = precedingSlotRow;
				}
				precedingSlotStart = tid;
				precedingSlotRow = sharedSlotRow;
			}
			delta = tid - precedingSlotStart;
		}
		scanDeltas[tid] = delta;
	}
	outputSlots[numOutputSlots] = curSharedMemSlot - 1;
	includedRowIndices[numOutputSlots++] = precedingSlotRow;

	// set deltaX to the second row
	// set deltaY to the third row
	// set outputSlots to the fourth row
	for(int tid(0), i(0); tid < WarpSize; ++tid, i += valuesPerThread) {
		warpColIndices[i + 1] |= scanDeltas[tid]<< 26;
		warpColIndices[i + 2] |= scanDeltas[WarpSize + tid]<< 25;
		if(tid < numOutputSlots) {
			warpColIndices[i + 1] |= SerializeRow;
			warpColIndices[i + 3] |= outputSlots[tid]<< 25;
		}
	}
	return numOutputSlots;
}

// Return the number of elements of the input consumed by the encoding as first.
// Return the output count as second.
template<typename T>
StreamResult EncoderBuilder::ConsumeAndWrite(const T* sparseValues,
	const int* colIndices, const int* rowIndices, int nz, T* sparseValuesOut, 
	uint* colIndicesOut) {

	assert(nz > 0);

	// We can start a block at any row, as the output index is encoded. This
	// lets us encode matrices with many consecutive zeros with negligible
	// waste.
	int firstGroupRow = rowIndices[0];
	int lastGroupRow = firstGroupRow + WarpSize - 1;

	StreamResult result;
	result.consumed = 0;

	int row = rowIndices[0];
	int col = colIndices[0];
	for(int i(0); i < groupSize; ++i) {
		T value;
		if(result.consumed < nz) {
			int row2 = rowIndices[result.consumed];

			if(row2 <= lastGroupRow) {
				row = row2;
				col = colIndices[result.consumed];
				value = sparseValues[result.consumed];
				++result.consumed;
			} else
				// insert the previous row and don't consume this input
				value = 0;
		} else
			// fill out the remaining elements with copies of the preceding one
			value = 0;

		// cache the row and column indices
		rowIndices2[i] = row;
		colIndices2[i] = col;
		sparseValuesOut[transposeIndices[i]] = value;
	}
	result.outputCount = ProcessWarp();

	// transpose the column indices
	for(int i(0); i < groupSize; ++i)
		colIndicesOut[transposeIndices[i]] = warpColIndices[i];
	
	nzCount += result.consumed;
	
	return result;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// EncoderFeeder is an adapter to feed EncoderBuilder from a variety of inputs.
// It also manages output.

template<typename T>
struct EncoderFeeder {

	void Init(int height_, int width_, int valuesPerThread_, int sizeHint);

	int Process(const T* sparse, const int* col, const int* row, int nz);

	void Finalize(std::auto_ptr<EncodedMatrixData<T> >* ppMatrixData);

	EncoderBuilder builder;
	int height;
	int width;
	int maxFragments;
	int outputSize;

	std::vector<T> sparseValuesOut;
	std::vector<uint> colIndicesOut;
	std::vector<uint> rowIndicesOut;
	std::vector<uint> outputCountsOut;
};

template<typename T>
void EncoderFeeder<T>::Init(int height_, int width_, int valuesPerThread_,
	int sizeHint) {

	builder.Init(valuesPerThread_);
	height = height_;
	width = width_;
	maxFragments = 0;
	outputSize = 0;

	sparseValuesOut.reserve((int)(1.01 * sizeHint));
	colIndicesOut.reserve((int)(1.01 * sizeHint));
	outputCountsOut.resize(height + 1);
}

template<typename T>
int EncoderFeeder<T>::Process(const T *sparse, const int *col, const int *row, 
	int nz) {

	size_t current = sparseValuesOut.size();
	sparseValuesOut.resize(current + builder.groupSize);
	colIndicesOut.resize(current + builder.groupSize);
	StreamResult result = builder.ConsumeAndWrite(sparse, col, row, nz, 
		&sparseValuesOut[0] + current, &colIndicesOut[0] + current);

	// append the row index 
	rowIndicesOut.push_back(outputSize);
	outputSize += result.outputCount;
	
	// fix me:
	// the outputs are no longer sequential.
	for(int i(0); i < result.outputCount; ++i)
		maxFragments = std::max<int>(
			++outputCountsOut[builder.includedRowIndices[i]], maxFragments);

	return result.consumed;
}	


template<typename T>
void EncoderFeeder<T>::Finalize(std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	std::auto_ptr<EncodedMatrixData<T> > matrix(new EncodedMatrixData<T>);
	matrix->height = height;
	matrix->width = width;
	matrix->valuesPerThread = builder.valuesPerThread;
	matrix->nz = builder.nzCount;
	matrix->nz2 = (int)sparseValuesOut.size();
	matrix->numGroups = (int)rowIndicesOut.size();
	matrix->outputSize = outputSize;

	matrix->sparseValues.swap(sparseValuesOut);
	matrix->colIndices.swap(colIndicesOut);
	matrix->rowIndices.swap(rowIndicesOut);
	matrix->outputIndices.swap(outputCountsOut);
	
	// calculate packedSizeShift

	int groupSizeBits = FindMaxBit(outputSize) + 1;
	int fragmentBits = 32 - groupSizeBits;
	if((1<< fragmentBits) <= maxFragments) {
		matrix->packedSizeShift = 0;

		// run an exclusive scan without packing bits
		int prevCount = matrix->outputIndices[0];
		matrix->outputIndices[0] = 0;
		for(int i(1); i <= height; ++i) {
			int count = matrix->outputIndices[i];
			matrix->outputIndices[i] = matrix->outputIndices[i - 1] + prevCount;
			prevCount = count;
		}

	} else {
		matrix->packedSizeShift = groupSizeBits;

		// run an exclusive scan with packing bits
	//	int mask = (1<< groupSizeBits) - 1;
		int offset = matrix->outputIndices[0];
		matrix->outputIndices[0] <<= groupSizeBits;
		for(int i(1); i < height; ++i) {
			int count = matrix->outputIndices[i];
			matrix->outputIndices[i] = offset | (count<< groupSizeBits);
			offset += count;
		}
	}
	*ppMatrix = matrix;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// EncodeMatrixDeinterleaved

template<typename T>
void EncodeMatrixDeinterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const T* sparse, 
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	EncoderFeeder<T> feeder;
	std::vector<int> rowIndices;
	
	if(SPARSE_INPUT_CSR == input) {
		rowIndices.resize(nz);

		// Expand the row indices into a full array.
		for(int r(0); r < height; ++r) {
			int begin = row[r];
			int end = row[r + 1];
			std::fill(&rowIndices[0] + begin, &rowIndices[0] + end, r);
		}
		row = &rowIndices[0];
	} 
	feeder.Init(height, width, valuesPerThread, nz);
	while(nz) {
		int consumed = feeder.Process(sparse, col, row, nz);
		sparse += consumed;
		col += consumed;
		row += consumed;
		nz -= consumed;
	}

	feeder.Finalize(ppMatrix);
}

template void EncodeMatrixDeinterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const float* sparse,
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<float> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const double* sparse,
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<double> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const cfloat* sparse,
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<cfloat> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const cdouble* sparse,
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<cdouble> >* ppMatrix);


////////////////////////////////////////////////////////////////////////////////
// EncodeMatrixInterleaved

template<typename T>
void EncodeMatrixInterleaved(int height, int width, int valuesPerThread, 
	sparseInput_t input, int nz, const void* sparse, const int* row, 
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	EncoderFeeder<T> feeder;
	feeder.Init(height, width, valuesPerThread, nz);
	int groupSize = feeder.builder.groupSize;

	if(SPARSE_INPUT_CSR == input) {
		const CSRElement<T>* csr = static_cast<const CSRElement<T>*>(sparse);
		std::vector<int> rowIndices(nz);

		// Expand the row indices into a full array.
		for(int r(0); r < height; ++r) {
			int begin = row[r];
			int end = row[r + 1];
			std::fill(&rowIndices[0] + begin, &rowIndices[0] + end, r);
		}
		row = &rowIndices[0];

		std::vector<T> sparseValues(groupSize);
		std::vector<int> colIndices(groupSize);
	
		while(nz) {
			int count = std::min(nz, groupSize);
			
			for(int i(0); i < count; ++i) {
				sparseValues[i] = csr[i].value;
				colIndices[i] = csr[i].col;
			}
			int consumed = feeder.Process(&sparseValues[0], &colIndices[0], row,
				nz);
			csr += consumed;
			row += consumed;
			nz -= consumed;
		}
	} else if(SPARSE_INPUT_COO == input) {
		const COOElement<T>* coo = static_cast<const COOElement<T>*>(sparse);
		std::vector<T> sparseValues(groupSize);
		std::vector<int> colIndices(groupSize);
		std::vector<int> rowIndices(groupSize);


		while(nz) {
			int count = std::min(nz, groupSize);
			
			for(int i(0); i < count; ++i) {
				sparseValues[i] = coo[i].value;
				colIndices[i] = coo[i].col;
				rowIndices[i] = coo[i].row;
			}
			int consumed = feeder.Process(&sparseValues[0], &colIndices[0],
				&rowIndices[0], nz);
			coo += consumed;
			nz -= consumed;
		}
	}
	feeder.Finalize(ppMatrix);
}

template void EncodeMatrixInterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<float> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<double> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<cfloat> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<cdouble> >* ppMatrix);

*/

const int STORE_FLAG = 1<< 25;

template<typename T>
void EncodeMatrixCOO(int height, int width, int vt, int nz, 
	const int* rowIndicesIn, const int* colIndicesIn, const T* valuesIn,
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	std::auto_ptr<EncodedMatrixData<T> > m(new EncodedMatrixData<T>);
	m->height = height;
	m->width = width;
	m->valuesPerThread = vt;
	m->nz = nz;
	m->colIndices.reserve((int)(1.02 * nz));
	m->sparseValues.reserve((int)(1.02 * nz));
	m->rowIndices.resize(height + 1);

	std::vector<int> cols(count);
	std::vector<T> vals(count);
	
	int count = WarpSize * vt;

	int curIn = 0;
	int curOut = 0;

	int outputIndex = 0;

	int lastStoredRow = 0;
	int lastStoredCount = 0;

	while(cur < nz) {

		////////////////////////////////////////////////////////////////////////
		// Read in the rows, cols, and values used in this block.

		int end = std::min(nz - cur, count);
		int firstRow = *rowIndicesIn;
		int prevGroupRow = prevRow;

		for(int i(0); i < count; ++i) {
				
			int row = 0x7fffffff;
			int col = 0;
			T val = 0;

			if(i < end) row = rowIndicesIn[i];
			if(row >= firstRow + WarpSize) end = i;
			if(i < end) {
				col = colIndicesIn[i];
				val = valuesIn[i];
				prevGroupRow = row;
			} else
				row = prevGroupRow;
			
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
			int* c = &cols[vt * lane];
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
				} else
					sharedRows[storeSlots + laneCount++] = curRow;
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
			cols[index + 2] |= distances[32 + lane];
			cols[index + 3] |= rowSumOffsets[lane]<< 26;
		}

		
		///////////////////////////////////////////////////////////////////////
		// Transpose the column indices and sparse values into 
		// EncodedMatrixData.

		m->colIndices.resize(curOut + WarpSize);
		m->sparseValues.resize(curOut + WarpSize);
		int* c = &m->colIndices[curOut];
		T* v = &m->sparseValues[curOut];

		for(int j(0), i(0); j < vt; ++j)
			for(int lane(0); lane < WarpSize; ++lane, ++i) {
				int i2 = vt * lane + j;
				c[i2] = cols[i];
				v[i2] = vals[i];
			}


		///////////////////////////////////////////////////////////////////////
		// Prepare the output indices and row indices.

		m->outputIndices.push_back(outputIndex);
		outputIndex += encountered;

		for(int i(0); i < encountered - 1; ++i) {
			// store prevRow

			int row = sharedRows[i];
			if(row == prevRow) {
				
			}
		}








		rowIndicesIn += end;
		colIndicesIn += end;
		valuesIn += end;
		cur += end;
	}
}

