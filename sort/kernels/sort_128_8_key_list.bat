nvcc --ptx -D SCATTER_TRANSACTION_LIST -D NUM_THREADS=128 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_NONE -arch=compute_20 -code=sm_20 -o ../ptx/sort_128_8_key_list.ptx sortgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ../cubin/sort_128_8_key_list.cubin ../ptx/sort_128_8_key_list.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_128_8_key_list.cubin > ../isa/sort_128_8_key_list.isa


