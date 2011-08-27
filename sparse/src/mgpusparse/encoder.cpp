
#include "engine.h"
#include <cassert>

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
	int warpSize, valuesPerThread, groupSize;
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

	void Init(int warpSize_, int valuesPerThread_) {
		precedingRow = 0;
		warpSize = warpSize_;
		valuesPerThread = valuesPerThread_;
		groupSize = warpSize * valuesPerThread;
		nzCount = 0;
		warpColIndices.resize(groupSize);
		includedRowIndices.resize(warpSize);
		sharedMemSlots.resize(2 * warpSize);
		scanDeltas.resize(2 * warpSize);
		outputSlots.resize(warpSize);
		rowIndices2.resize(groupSize);
		colIndices2.resize(groupSize);

		transposeIndices.resize(groupSize);
		for(int warp(0), i(0); warp < warpSize; ++warp)
			for(int val(0); val < valuesPerThread; ++val, ++i)
				// scatter from i to the transposed index
				transposeIndices[i] = (i % valuesPerThread) * warpSize +
					(i / valuesPerThread);
	}
};

int EncoderBuilder::ProcessWarp() {

	// Loop through each value in row-major order and build the colIndex flags.
	int curSharedMemSlot = 0;
	int lastFlagCount = 0;
	for(int tid(0), i(0); tid < warpSize; ++tid) {
		int curThreadRow = -1;
		int lastFlagCount = 0;
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
	int curScanRow = -1;
	int precedingSlotRow = -1;
	int precedingSlotStart = -1;
	int numOutputSlots = 0;
	for(int tid(0); tid < 2 * warpSize; ++tid) {
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
	for(int tid(0), i(0); tid < warpSize; ++tid, i += valuesPerThread) {
		warpColIndices[i + 1] |= scanDeltas[tid]<< 26;
		warpColIndices[i + 2] |= scanDeltas[warpSize + tid]<< 25;
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
	int lastGroupRow = firstGroupRow + warpSize - 1;

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

	void Init(int height_, int width_, int warpSize_, int valuesPerThread_, int sizeHint,
		bool allocSparse, bool allowRow, bool allocCol);

	int Process(const T* sparse, const int* row, const int* col, int nz);

	typedef size_t(SPARSEAPI*FP)(size_t, T*, int*, int*, void*);
	void FeedStream(FP fp, void* cookie, std::auto_ptr<EncodedMatrixData<T> >* ppMatrix);

	void Finalize(std::auto_ptr<EncodedMatrixData<T> >* ppMatrixData);

	void Advance(int count) {
		memmove(&sparseFrame[0], &sparseFrame[0] + builder.groupSize - count, sizeof(T) * count);
		memmove(&rowFrame[0], &rowFrame[0] + builder.groupSize - count, sizeof(int) * count);
		memmove(&colFrame[0], &colFrame[0] + builder.groupSize - count, sizeof(int) * count);
	}

	EncoderBuilder builder;
	int height;
	int width;
	int maxFragments;
	int outputSize;

	std::vector<T> sparseValuesOut;
	std::vector<uint> colIndicesOut;
	std::vector<uint> rowIndicesOut;
	std::vector<uint> outputCountsOut;

	std::vector<T> sparseFrame;
	std::vector<int> rowFrame;
	std::vector<int> colFrame;
};

template<typename T>
void EncoderFeeder<T>::Init(int height_, int width_, int warpSize_, int valuesPerThread_,
	int sizeHint, bool allocSparse, bool allocRow, bool allocCol) {

	builder.Init(warpSize_, valuesPerThread_);
	height = height_;
	width = width_;
	maxFragments = 0;
	outputSize = 0;

	sparseValuesOut.reserve((int)(1.01 * sizeHint));
	colIndicesOut.reserve((int)(1.01 * sizeHint));
	outputCountsOut.resize(height + 1);

	if(allocSparse) sparseFrame.resize(builder.groupSize);
	if(allocRow) rowFrame.resize(builder.groupSize);
	if(allocCol) colFrame.resize(builder.groupSize);
}

template<typename T>
int EncoderFeeder<T>::Process(const T *sparse, const int *col, const int *row, int nz) {

	if(!sparse) sparse = &sparseFrame[0];
	if(!col) col = &colFrame[0];
	if(!row) row = &rowFrame[0];

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
	matrix->warpSize = builder.warpSize;
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
		int mask = (1<< groupSizeBits) - 1;
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
void EncodeMatrixDeinterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const T* sparse, 
	const int* col, const int* row, 
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	EncoderFeeder<T> feeder;
	
	if(SPARSE_INPUT_CSR == input) {
		feeder.Init(height, width, warpSize, valuesPerThread, nz, false, true, false);
		int curRow = 0;
		while(nz) {
			// advance index until the current row is found
			int count = std::min(nz, warpSize);
			for(int i(0); i < count; ) {
				if(i + feeder.builder.nzCount >= row[curRow]) ++curRow;
				else feeder.rowFrame[i++] = curRow;
			}
			int consumed = feeder.Process(sparse, 0, col, nz);
			sparse += consumed;
			col += consumed;
			nz -= consumed;
		}
	} else if(SPARSE_INPUT_COO == input) {
		feeder.Init(height, width, warpSize, valuesPerThread, nz, false, false, false);
		while(nz) {
			int consumed = feeder.Process(sparse, col, row, nz);
			sparse += consumed;
			col += consumed;
			row += consumed;
			nz -= consumed;
		}
	}

	feeder.Finalize(ppMatrix);
}

template void EncodeMatrixDeinterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const float* sparse,
	const int* col, const int* row, std::auto_ptr<EncodedMatrixData<float> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const double* sparse,
	const int* col, const int* row, std::auto_ptr<EncodedMatrixData<double> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const cfloat* sparse,
	const int* col, const int* row, std::auto_ptr<EncodedMatrixData<cfloat> >* ppMatrix);
template void EncodeMatrixDeinterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const cdouble* sparse,
	const int* col, const int* row, std::auto_ptr<EncodedMatrixData<cdouble> >* ppMatrix);


///////////////////////////////////////////////////////////////////////////////////////////////////
// EncodeMatrixInterleaved

template<typename T>
void EncodeMatrixInterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const void* sparse,
	const int* row, std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	EncoderFeeder<T> feeder;
	feeder.Init(height, width, warpSize, valuesPerThread, nz, true, true, true);

	if(SPARSE_INPUT_CSR == input) {
		const CSRElement<T>* csr = static_cast<const CSRElement<T>*>(sparse);
		int curRow = 0;
		while(nz) {
			// advance index until the current row is found
			int count = std::min(nz, warpSize);
			for(int i(0); i < count; ) {
				if(i + feeder.builder.nzCount >= row[curRow]) ++curRow;
				else {
					feeder.sparseFrame[i] = csr[i].value;
					feeder.rowFrame[i] = curRow;
					feeder.colFrame[i] = csr[i].col;
					++i;
				}			
			}
			int consumed = feeder.Process(0, 0, 0, nz);
			csr += consumed;
		}
	} else if(SPARSE_INPUT_COO == input) {
		const COOElement<T>* coo = static_cast<const COOElement<T>*>(sparse);
		while(nz) {
			for(int i(0); i < std::min(feeder.builder.groupSize, nz); ++i) {
				feeder.sparseFrame[i] = coo[i].value;
				feeder.rowFrame[i] = coo[i].row;
				feeder.colFrame[i] = coo[i].col;
			}
			int consumed = feeder.Process(0, 0, 0, nz);
			coo += consumed;
			nz -= consumed;
		}
	}
	feeder.Finalize(ppMatrix);
}

template void EncodeMatrixInterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<float> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<double> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<cfloat> >* ppMatrix);
template void EncodeMatrixInterleaved(int height, int width, int warpSize,
	int valuesPerThread, sparseInput_t input, int nz, const void* data,
	const int* row, std::auto_ptr<EncodedMatrixData<cdouble> >* ppMatrix);


///////////////////////////////////////////////////////////////////////////////////////////////////
// EncodeMatrixStream

template<typename T>
void EncodeMatrixStream(int height, int width, int warpSize, int valuesPerThread,
	int sizeHint, int(SPARSEAPI*fp)(int, T*, int*, int*, void*), void* cookie,
	std::auto_ptr<EncodedMatrixData<T> >* ppMatrix) {

	EncoderFeeder<T> feeder;
	feeder.Init(height, width, warpSize, valuesPerThread, sizeHint, true, true, true);
	int leftover = 0;
	int readSize = 0;
	do {
		int groupSize = feeder.builder.groupSize;
		if(leftover) feeder.Advance(leftover);

		readSize = feeder.builder.groupSize - leftover;
		int nz = fp((size_t)readSize, &feeder.sparseFrame[0] + leftover,
			&feeder.rowFrame[0] + leftover, &feeder.colFrame[0] + leftover, cookie);
		readSize += nz;
		int consumed = feeder.Process(0, 0, 0, readSize);
	} while(readSize == feeder.builder.groupSize);

	feeder.Finalize(ppMatrix);
}


template void EncodeMatrixStream(int height, int width, int warpSize, 
	int valuesPerThread, int sizeHint, sparseStreamReal4_fp fp, void* cookie,
	std::auto_ptr<EncodedMatrixData<float> >* ppMatrix);
template void EncodeMatrixStream(int height, int width, int warpSize, 
	int valuesPerThread, int sizeHint, sparseStreamReal8_fp fp, void* cookie,
	std::auto_ptr<EncodedMatrixData<double> >* ppMatrix);
template void EncodeMatrixStream(int height, int width, int warpSize,
	int valuesPerThread, int sizeHint, sparseStreamComplex4_fp fp, void* cookie,
	std::auto_ptr<EncodedMatrixData<cfloat> >* ppMatrix);
template void EncodeMatrixStream(int height, int width, int warpSize,
	int valuesPerThread, int sizeHint, sparseStreamComplex8_fp fp, void* cookie,
	std::auto_ptr<EncodedMatrixData<cdouble> >* ppMatrix);

