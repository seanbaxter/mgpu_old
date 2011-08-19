nvcc -arch=compute_20 -code=sm_20 --ptx -o globalscan.ptx globalscan.cu
IF %ERRORLEVEL% EQU 0 ptxas -v -arch=sm_20 -o globalscan.cubin globalscan.ptx
IF %ERRORLEVEL% EQU 0 cuobjdump -sass globalscan.cubin > globalscan.isa
