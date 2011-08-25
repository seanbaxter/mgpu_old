
#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>

const int NumBuckets = 16;
const int NumValues = 32 * 16;
const int WarpSize = 16;
const int ValsPerSegment = 16;
const bool EmitHTML = true;

std::tr1::mt19937 mt19937;

std::vector<int> keys(NumValues);
std::tr1::uniform_int<int> r1(0, NumBuckets - 1);
std::tr1::uniform_int<int> r2(0, ValsPerSegment - 1);

std::vector<int> counts(NumBuckets), gather(NumBuckets), scatter(NumBuckets);


const char* Ticks = 
"|               |               |               |               |               |";
const char* Marks = 
"0              16              32              48              64              80";

void PrintArray(const int* vals, int count, int valsPerLine, int spacing,
	int leadingSpaces) {
	for(int i(0); i < count; ++i) {
		if(0 == (i % valsPerLine)) {
			if(i) printf("\n");
			for(int j(0); j < leadingSpaces; ++j)
				printf(" ");
		} else if(0 == (i % spacing)) 
			printf(" ");
		
		printf("%x", vals[i]);
	}
	printf("\n");
}

int PrintTransactionsSimple(int digit, int gather, int scatter, int count,
	int* prevTrans, int transCol) {

	// Print the radix digit and space up to the first scattered value.
	printf("%x:  ", digit);
	for(int i(0); i < scatter; ++i) 
		printf(" ");
		
	int prevSegment = -1;
	int prevWarp = -1;
	int transCount = 0;
	for(int i(0); i < count; ++i) {
		int curSegment = (scatter + i) / ValsPerSegment;
		int curWarp = (gather + i) / WarpSize;
		if((prevSegment != curSegment) || (prevWarp != curWarp)) {
			if(EmitHTML) {
				if(i) printf("</span>");
				printf("<span class=\"%s\">", *prevTrans ? "blue" : "red");
				*prevTrans ^= 1;
			}
			prevSegment = curSegment;
			prevWarp = curWarp;
			++transCount;
		}

		printf("%x", (WarpSize - 1) & (gather + i));
	}
	if(count && EmitHTML) printf("</span>");
	
	int offset = 4 + scatter + count;
	while(offset < transCol) { printf(" "); ++offset; }
	if(EmitHTML)
		printf("<strong>%2d</strong> vals, <strong>%d</strong> trans\n",
			count, transCount);
	else
		printf("%2d vals, %d trans\n", count, transCount);


	return transCount;
}

int PrintTransactionsAligned(int digit, int gather, int scatter, int count,
	 int* prevTrans, int transCol) {

	// Print the radix digit and space up to the first scattered value.
	printf("%x:  ", digit);
	for(int i(0); i < scatter; ++i) 
		printf(" ");
		
	int seg = -1;
	int segStart = -1;
	int transCount = 0;
	for(int i(0); i < count; ++i) {
		int curSeg = (scatter + i) / WarpSize;
		if(seg != curSeg) {
			if(EmitHTML) {
				if(i) printf("</span>");
				printf("<span class=\"%s\">", *prevTrans ? "blue" : "red");
				*prevTrans ^= 1;
			}
			seg = curSeg;
			segStart = i;
			++transCount;
		}

		printf("%x", i - segStart);
	}
	if(count && EmitHTML) printf("</span>");
	
	int offset = 4 + scatter + count;
	while(offset < transCol) { printf(" "); ++offset; }
	if(EmitHTML)
		printf("<strong>%2d</strong> vals, <strong>%d</strong> trans\n",
			count, transCount);
	else
		printf("%2d vals, %d trans\n", count, transCount);

	return transCount;
}

int PrintScatter(bool aligned) {

	printf("*** %s scatter:\n", aligned ? "Aligned" : "Simple");
	printf("    %s\n    %s\n", Marks, Ticks);
	int prevTrans = 0;
	int transTotal = 0;
	for(int i(0); i < NumBuckets; ++i) {
		int transCount;
		if(aligned)
			transCount = PrintTransactionsAligned(i, gather[i], scatter[i],
				counts[i], &prevTrans, 69);
		else
			transCount = PrintTransactionsSimple(i, gather[i], scatter[i],
				counts[i], &prevTrans, 69);
		transTotal += transCount;
	}
	printf("    %s\n    %s\n", Ticks, Marks);
	printf(EmitHTML ?
		"*** Total transactions: <strong>%2d</strong>\n" :
		"*** Total transactions: %2d\n", transTotal);

	return transTotal;
}


int main() {

	printf("Sorting %d values with a warp and segment size of %d.\n",
		NumValues, WarpSize);
	printf("Number of radix digits is %d.\n\n", NumBuckets);

	for(int i(0); i < NumValues; ++i) {
		keys[i] = r1(mt19937);
		++counts[keys[i]];
	}

	printf("sequence:\n");
	PrintArray(&keys[0], NumValues, 64, 16, 4);
	std::sort(keys.begin(), keys.end());
	printf("\n");

	printf("sorted:\n");
	PrintArray(&keys[0], NumValues, 64, 16, 4);
	printf("\n");

	for(int i(0); i < NumBuckets; ++i) {
		if(i) gather[i] = gather[i - 1] + counts[i - 1];
		scatter[i] = r2(mt19937);

		if(EmitHTML) {
			printf("\t* digit <strong>%x</strong>   count <strong>%2d</strong>"
				"   gather <strong>%3d</strong> (lane <strong>%x</strong>)"
				"   scatter lane <strong>%x</strong>\n", i, counts[i],
				gather[i], 0xf & gather[i], scatter[i]);

		} else {
			printf("\t* digit %x   count %2d   gather %3d (lane %x)"
				"   scatter lane %x\n", i, counts[i], gather[i],
				0xf & gather[i], scatter[i]);
		}
	}
	printf("\n");
	for(int i(0); i < 85; ++i) printf("-");
	printf("\n");

	int simpleTrans = PrintScatter(false);

	printf("\n");
	for(int i(0); i < 85; ++i) printf("-");
	printf("\n");

	int alignedTrans = PrintScatter(true);

	printf("\n");
	for(int i(0); i < 85; ++i) printf("-");
	printf("\n");

	double ratio = (double)alignedTrans / simpleTrans;
	printf(EmitHTML ?
		"*** Aligned:Simple transaction ratio: <strong>%3.2lf</strong>\n" : 
		"*** Aligned:Simple transaction ratio: %3.2lf\n", ratio);
}
