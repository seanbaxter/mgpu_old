nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_DOUBLE -D USE_COMPLEX -o ..\cubin\spmxv_cdouble.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_cdouble.cubin > ..\isa\spmxv_cdouble.isa

