VPATH = ../../src/mgpuscan ../../../util/

debug: CPPFLAGS = -I /usr/local/cuda/include -g -O0
release: CPPFLAGS = -I /usr/local/cuda/include -O3

debug: TARGET = debug
release: TARGET = release

debug: ../debug/libmgpuscan.a
release: ../release/libmgpuscan.a

../debug/libmgpuscan.a : \
	mgpuscan.cpp \
	util.h \
	cucpp.h \
	cucpp.cpp
	-mkdir ../$(TARGET)
	g++ $(CPPFLAGS) -o $@ $<

