nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_COMPLEX -D USE_FLOAT_DOUBLE -o ..\cubin\spmxv_cfloat_double.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_cfloat_double.cubin > ..\isa\spmxv_cfloat_double.isa

