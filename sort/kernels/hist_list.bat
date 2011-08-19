nvcc --ptx -D INCLUDE_TRANSACTION_LIST -arch=compute_20 -code=sm_20 -o ..\ptx\hist_list.ptx histgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ..\cubin\hist_list.cubin ..\ptx\hist_list.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\hist_list.cubin > ..\isa\hist_list.isa

