nvcc --ptx -arch=compute_20 -code=sm_20 -o ..\ptx\hist_simple.ptx histgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ..\cubin\hist_simple.cubin ..\ptx\hist_simple.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\hist_simple.cubin > ..\isa\hist_simple.isa

