nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_DOUBLE -o ..\cubin\spmxv_double.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_double.cubin > ..\isa\spmxv_double.isa

