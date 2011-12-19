#pragma once

#include "../../../inc/mgpubwt.h"
#include "../../../inc/mgpusort.hpp"
#include "../../../util/cucpp.h"
#include <memory>
#include <sstream>
#include <algorithm>

typedef unsigned char byte;

const int BWTGatherThreads = 512;
const int BWTCompareThreads = 512;
const int TinyBlockCutoff = 3000;

// Sort an interval of string indices. symbols should point to the extended
// symbol array, adjusted for the number of bytes that are already confirmed
// matches. count2 should be count minus the number of confirmed matched bytes.
void QSortStrings(int* left, int* right, const byte* symbols, int count2);
