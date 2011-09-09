vpath %.h ../../src/simpletest ../../../util
vpath %.cpp ../../src/simpletest ../../../util

INC = cucpp.h util.h
SRC = simpletest.cpp cucpp.cpp

.PHONY: release
.PHONY: debug

release: CPPFLAGS1 = -Wall -I /usr/local/cuda/include -c -O3
release: CPPFLAGS2 = -Wall -lcuda
debug: CPPFLAGS1 = -Wall -I /usr/local/cuda/include -c -g -O0
debug: CPPFLAGS2 = -Wall -l cuda

release: ../release/simpletest
debug: ../debug/simpletest

../release/simpletest : release/simpletest.o release/cucpp.o
	@mkdir -p ../release
	g++ $(CPPFLAGS2) -o $@ $^ ../release/libmgpusort.a

release/%.o : %.cpp $(INC)
	@mkdir -p release
	g++ $(CPPFLAGS1) -o $@ $<

../debug/simpletest : debug/simpletest.o debug/cucpp.o
	@mkdir -p ../debug
	g++ $(CPFLAGS2) -o $@ $^ ../debug/libmgpusort.a

debug/%.o : %.cpp $(INC)
	@mkdir -p debug
	g++ $(CPPFLAGS1) -o $@ $<

.PHONY: clean
clean:
	@-rm release/*.o
	@-rm ../release/simpletest
	@-rm debug/*.o
	@-rm ../debug/simpletest

