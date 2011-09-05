nvcc --cubin -arch=compute_20 -code=sm_20 -Xptxas=-v -o maxindex.cubin maxindex.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass maxindex.cubin > maxindex.isa