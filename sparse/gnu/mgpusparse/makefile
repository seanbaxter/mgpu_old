vpath %.h ../../src/mgpusparse ../../../inc/ ../../../util
vpath %.cpp ../../src/mgpusparse ../../../util

INC = mgpusparse.h engine.h cucpp.h util.h
SRC = mgpusparse.cpp engine.cpp encoder.cpp cucpp.cpp 

.PHONY: release
.PHONY: debug


release: CPPFLAGS = -Wall -I /usr/local/cuda/include -c -O3
debug: CPPFLAGS = -Wall -I /usr/local/cuda/include -c -g -O0
release : ../release/libmgpusparse.a $(INC) $(SRC)
debug : ../debug/libmgpusparse.a $(INC) $(SRC)


../release/libmgpusparse.a : \
	release/mgpusparse.o \
	release/engine.o \
	release/encoder.o \
	release/cucpp.o
	@mkdir -p ../release
	ar rv $@ $^

release/%.o : %.cpp $(INC)
	@mkdir -p release
	g++ $(CPPFLAGS) -o $@ $<

../debug/libmgpusparse.a : \
	debug/mgpusparse.o \
	debug/engine.o \
	debug/encoder.o \
	debug/cucpp.o
	@mkdir -p ../debug
	ar rv $@ $^

debug/%.o : %.cpp $(INC)
	@mkdir -p debug
	g++ $(CPPFLAGS) -o $@ $<


.PHONY: clean
clean:
	@-rm debug/*.o
	@-rm ../debug/libmgpusparse.a
	@-rm release/*.o
	@-rm ../release/libmgpusparse.a
