#include <cstdio>
#include <random>
#include <vector>
#include <functional>
#include <algorithm>

const bool PrintHTML = true;

typedef std::pair<int, int> IntPair;

int DivUp(int x, int y) {
	return (x + y - 1) / y;
}
 
void PrintDigitHTML(int digit, int place, bool html) {
	if(html) printf("<span class=\"red\">");
	printf("%d", (digit / place) % 10);
	if(html) printf("</span>");
}

void PrintNumberHTML(int x, int place) {
	if(PrintHTML) {
		PrintDigitHTML(x, 100, 100 == place);
		PrintDigitHTML(x, 10, 10 == place);
		PrintDigitHTML(x, 1, 1 == place);
		printf("  ");
	} else
		printf("%03d  ", x); 
}

void PrintArray(const int* values, int count, int place) {
	int lines = DivUp(count, 16);
	for(int line(0); line < lines; ++line) {
		int cols = std::min(count, 16);
		count -= cols;

		for(int col(0); col < cols; ++col)
			PrintNumberHTML(values[col], place);
		printf("\n");
		values += cols;
	}
}

int GetDigit(int x, int place) {
	return (x / place) % 10;
}

// Returns the digit with the k'th element.
IntPair ProcessDigit(const int* source, int count, int place, int k, 
	std::vector<int>& dest) {

	for(int i(0); i < 80; ++i)
		printf("-");
	printf("\n%d's place:\n\n", place);

	// Build the histogram by counting the number of elements with each digit.
	int hist[10] = { 0 };
	for(int i(0); i < count; ++i)
		++hist[GetDigit(source[i], place)];

	// Scan the histogram and find which radix digit k maps into.
	int scan[10];
	int kDigit;
	scan[0] = 0;
	for(int i(1); i < 10; ++i) {
		scan[i] = scan[i - 1] + hist[i - 1];
		if((k >= scan[i]) && (k < (scan[i] + hist[i]))) 
			kDigit = i;
	}

	// Print the counts and scans.
	printf("Frequency and scan of %d's digit:\n", place);
	for(int i(0); i < 10; ++i) {
		printf("%d: %3d   %3d   %c\n", i, hist[i], scan[i], 
			(kDigit == i) ? '*' : ' ');	
	}

	// Extract all the elements with a kDigit.
	int kCount = hist[kDigit];
	dest.resize(kCount);
	for(int i(0), j(0); i < count; ++i) {
		int x = source[i];
		if(kDigit == GetDigit(x, place))
			dest[j++] = x;
	}

	printf("\nValues after filter (%d values):\n", kCount);
	PrintArray(&dest[0], kCount, place / 10);
	printf("\n");

	// Return kDigit and scan[kDigit].
	return IntPair(kDigit, scan[kDigit]);
}

int main(int argc, char** argv) {
	std::tr1::mt19937 mt19937;
	std::tr1::uniform_int<int> r(0, 999);

	const int Count = 16 * 32;
	std::vector<int> values(Count);
	for(int i(0); i < Count; ++i)
		values[i] = r(mt19937);

	// Build the histogram.
	int digits1[10] = { 0 };
	for(int i(0); i < Count; ++i)
		++digits1[values[i] / 100];

	// std::sort(values.begin(), values.end());

	int k = 2 * Count / 3;
	printf("Searching for k = %d.\n\n", k);
	printf("Full sequence (%d) values:\n", Count);
	PrintArray(&values[0], Count, 100);
	printf("\n");


	// Run the search on the 100's digit.
	std::vector<int> hundreds;
	IntPair result100 = ProcessDigit(&values[0], Count, 100, k, hundreds);
	printf("k is in %d for 100s digit. Scan offset for this digit is %d.\n",
		result100.first, result100.second);
	printf("Adjusted k is %d - %d = %d.\n\n\n", k, result100.second,
		k - result100.second);
	k -= result100.second;

	// Run the search on the 10's digit.
	std::vector<int> tens;
	IntPair result10 = ProcessDigit(&hundreds[0], (int)hundreds.size(), 10,
		k, tens);
	printf("k is in %d for 10s digit. Scan offset for this digit is %d.\n",
		result10.first, result10.second);
	printf("Adjusted k is %d - %d = %d.\n\n\n", k, result10.second,
		k - result10.second);
	k -= result10.second;

	// Run the search on the 1's digit.
	std::vector<int> ones;
	IntPair result1 = ProcessDigit(&tens[0], (int)tens.size(), 1, k, ones);
	printf("k is in %d for 1s digit. Scan offset for this digit is %d.\n",
		result1.first, result1.second);
	printf("Adjusted k is %d - %d = %d.\n\n\n", k, result1.second, 
		k - result1.second);
	k -= result1.second;

	for(int i(0); i < 80; ++i)
		printf("-");
	printf("\nk'th smallest element is %d!\n", ones[k]);


}