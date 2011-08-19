nvcc --ptx -D SCATTER_SIMPLE -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_INDEX -arch=compute_20 -code=sm_20 -o ../ptx/sort_128_8_index_simple.ptx sortgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ../cubin/sort_128_8_index_simple.cubin ../ptx/sort_128_8_index_simple.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_128_8_index_simple.cubin > ../isa/sort_128_8_index_simple.isa


