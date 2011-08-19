nvcc -ptx -arch=compute_20 -code=sm_20 -o ..\ptx\count.ptx countgen.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o ..\cubin\count.cubin ..\ptx\count.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\count.cubin > ..\isa\count.isa

