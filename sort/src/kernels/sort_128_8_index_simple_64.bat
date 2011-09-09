nvcc -m=64 --cubin -D BUILD_64 -Xptxas=-v -D SCATTER_SIMPLE -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_INDEX -arch=compute_20 -code=sm_20 -o ../cubin64/sort_128_8_index_simple.cubin sortgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin64/sort_128_8_index_simple.cubin > ../isa64/sort_128_8_index_simple.isa


