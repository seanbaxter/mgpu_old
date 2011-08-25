nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ..\cubin\hist_simple.cubin histgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\hist_simple.cubin > ..\isa\hist_simple.isa

