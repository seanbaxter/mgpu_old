vpath %.h ../../src/sparsetest ../../src/support ../../../inc ../../../util
vpath %.cpp ../../src/sparsetest ../../src/support ../../../util

INC = cucpp.h util.h sparse.h mgpusparse.h mmio.h
SRC = cucpp.cpp csrdevice.cpp matrix.cpp mmio.cpp sparsetest.cpp

.PHONY: release
.PHONY: debug

release: CPPFLAGS1 = -Wall -I /usr/local/cuda/include -c -O3
release: CPPFLAGS2 = -Wall -L /usr/local/cuda/lib -lcuda -lcusparse -lcudart
debug: CPPFLAGS1 = -Wall -I /usr/local/cuda/include -c -g -O0
debug: CPPFLAGS2 = -Wall -L /usr/local/cuda/lib -lcuda -lcusparse -lcudart

release: ../release/sparsetest
debug: ../debug/sparsetest

../release/sparsetest : \
	release/sparsetest.o \
	release/cucpp.o \
	release/csrdevice.o \
	release/matrix.o \
	release/mmio.o
	@mkdir -p ../release
	g++ $(CPPFLAGS2) -o $@ $^ ../release/libmgpusparse.a

release/%.o : %.cpp $(INC)
	@mkdir -p release
	g++ $(CPPFLAGS1) -o $@ $<

../debug/sparsetest : \
	debug/sparsetest.o \
	debug/cucpp.o \
	debug/csrdevice.o \
	debug/matrix.o \
	debug/mmio.o
	@mkdir -p ../debug
	g++ $(CPPFLAGS2) -o $@ $^ ../debug/libmgpusparse.a

debug/%.o : %.cpp $(INC)
	@mkdir -p debug
	g++ $(CPPFLAGS1) -o $@ $<

.PHONY: clean
clean:
	@-rm release/*.o
	@-rm ../release/sparsetest
	@-rm debug/*.o
	@-rm ../debug/sparsetest

