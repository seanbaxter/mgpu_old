
# Rules for generating all the kernels of these types.
.PHONY: counts
.PHONY: hists
.PHONY: sorts

# Rules for detecting changes in .cu files effecting these kernels.
.PHONY: count
.PHONY: hist
.PHONY: sort


# Relative paths
#vpath %.isa ../isa
#vpath %.cubin ../cubin

ISA := ../isa
CUBIN = ../cubin

# Compiler options
NVCCFLAGS = --cubin -Xptxas=-v -arch=compute_20 -code=sm_20


T128 = -D NUM_THREADS=128
T256 = -D NUM_THREADS=256

INDEX = -D VALUE_TYPE_INDEX
KEY = -D VALUE_TYPE_NONE
SINGLE = -D VALUE_TYPE_SINGLE
MULTI = -D VALUE_TYPE_MULTI

LIST = -D SCATTER_TRANSACTION_LIST
SIMPLE = -D SCATTER_SIMPLE

SORTFLAGS = $(NVCCFLAGS) -D VALUES_PER_THREAD=8


# Generate all kernels.
all : \
	sorts hists counts


# Generate all .isa files
counts : \
	$(ISA)/count.isa

hists : \
	$(ISA)/hist_simple.isa
	
sorts : \
	$(ISA)/sort_128_8_index_simple.isa \
	$(ISA)/sort_128_8_key_simple.isa \
	$(ISA)/sort_128_8_single_simple.isa \
	$(ISA)/sort_128_8_multi_simple.isa \
	$(ISA)/sort_256_8_index_simple.isa \
	$(ISA)/sort_256_8_key_simple.isa \
	$(ISA)/sort_256_8_single_simple.isa \
	$(ISA)/sort_256_8_multi_simple.isa
	

# Generate all .isa files from a pattern with cuobjdump.
$(ISA)/%.isa : $(CUBIN)/%.cubin
	cuobjdump -sass $< > $@
	
	
# Define dependencies for count, hist, and sort
count : count.cu countgen.cu common.cu
hist : histgen.cu histogram.cu common.cu params.cu \
	hist1.cu hist2.cu hist3.cu hist3_64.cu
sort : common.cu params.cu sort.cu sortcommon.cu sortgen.cu \
	sortgeneric.cu sortscan1.cu sortscan2.cu sortscan3.cu sortstore.cu
	
	 
# Build count cubins
$(CUBIN)/count.cubin : count
	nvcc $(NVCCFLAGS) -o $@ countgen.cu
	
# Build hist cubins
$(CUBIN)/hist_simple.cubin : hist
	nvcc $(NVCCFLAGS) -o $@ histgen.cu

# Build sort cubins
$(CUBIN)/sort_128_8_index_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T128) $(INDEX) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_128_8_key_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T128) $(KEY) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_128_8_single_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T128) $(SINGLE) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_128_8_multi_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T128) $(MULTI) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_256_8_index_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T256) $(INDEX) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_256_8_key_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T256) $(KEY) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_256_8_single_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T256) $(SINGLE) $(SIMPLE) -o $@ sortgen.cu
	
$(CUBIN)/sort_256_8_multi_simple.cubin : sort
	nvcc $(SORTFLAGS) $(T256) $(MULTI) $(SIMPLE) -o $@ sortgen.cu
	