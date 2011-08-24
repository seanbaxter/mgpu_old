nvcc --cubin -Xptxas=-v -D SCATTER_SIMPLE -D NUM_THREADS=256 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_INDEX -arch=compute_20 -code=sm_20 -o ../cubin/sort_256_8_index_simple.cubin sortgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_256_8_index_simple.cubin > ../isa/sort_256_8_index_simple.isa


