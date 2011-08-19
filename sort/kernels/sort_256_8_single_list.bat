nvcc --ptx -D SCATTER_TRANSACTION_LIST -D NUM_THREADS=256 -D VALUES_PER_THREAD=8 -D VALUE_TYPE_SINGLE -arch=compute_20 -code=sm_20 -o ../ptx/sort_256_8_single_list.ptx sortgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ../cubin/sort_256_8_single_list.cubin ../ptx/sort_256_8_single_list.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort_256_8_single_list.cubin > ../isa/sort_256_8_single_list.isa


