vpath %.h ../../src/mgpusort/ ../../../inc/ ../../../util
vpath %.cpp ../../src/mgpusort/ ../../../util

INC = mgpusort.h sortdll.h cucpp.h util.h
SRC = mgpusort.cpp sorttables.cpp cucpp.cpp

.PHONY: release
.PHONY: debug

release: CPPFLAGS = -Wall -I /usr/local/cuda/include -c -O3
debug: CPPFLAGS = -Wall -I /usr/local/cuda/include -c -g -O0
release : ../release/libmgpusort.a $(INC) $(SRC)
debug : ../debug/libmgpusort.a $(INC) $(SRC)

../release/libmgpusort.a : release/mgpusort.o release/sorttables.o release/cucpp.o
	@mkdir -p ../release
	ar rv $@ $^

release/%.o : %.cpp $(INC)
	@mkdir -p release
	g++ $(CPPFLAGS) -o $@ $<

../debug/libmgpusort.a : debug/mgpusort.o debug/sorttables.o debug/cucpp.o
	@mkdir -p ../debug
	ar crv $@ $^

debug/%.o : %.cpp $(INC)
	@mkdir -p debug
	g++ $(CPPFLAGS) -o $@ $<

.PHONY: clean
clean:
	@-rm debug/*.o
	@-rm ../debug/libmgpusort.a
	@-rm release/*.o
	@-rm ../release/libmgpusort.a