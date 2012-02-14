REM nvcc -m=32 --ptx -Xptxas=-v -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_NONE -arch=compute_20 -code=sm_20 -o ../ptx/sortloop_128_8_key_simple.ptx sortloopgen.cu

nvcc -m=32 --cubin -Xptxas=-v -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_NONE -arch=compute_20 -code=sm_20 -o ../cubin/sortloop_128_8_key_simple.cubin sortloopgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sortloop_128_8_key_simple.cubin > ../isa/sortloop_128_8_key_simple.isa


