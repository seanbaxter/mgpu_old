CPPFLAGS = -Wall -I /usr/local/cuda/include -l cuda

SRC = \
	../../src/mgpusort/mgpusort.cpp \
	../../src/mgpusort/sorttables.cpp \
	../../../util/cucpp.cpp
INC = \
	../../../inc/mgpusort.h \
	../../src/mgpusort/sortdll.h \
	../../../util/cucpp.h \
	../../../util/util.h

debug : ../debug/libmgpuscan.o

../debug/libmgpuscan.o : $(SRC)
	-mkdir ../debug
	g++ $(CPPFLAGS) -c -o $@ $^

