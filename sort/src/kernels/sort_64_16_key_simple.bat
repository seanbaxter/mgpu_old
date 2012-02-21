nvcc --open64 -m=32 --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D NUM_THREADS=64 -D VALUES_PER_THREAD=16 -o ../cubin/sort_64_16_key_simple.cubin sort2.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_64_16_key_simple.cubin > ../isa/sort_64_16_key_simple.isa


