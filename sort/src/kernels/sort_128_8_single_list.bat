nvcc -m=32 --cubin -Xptxas=-v -D SCATTER_TRANSACTION_LIST -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_SINGLE -arch=compute_20 -code=sm_20 -o ../cubin/sort_128_8_single_list.cubin sortgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_128_8_single_list.cubin > ../isa/sort_128_8_single_list.isa


