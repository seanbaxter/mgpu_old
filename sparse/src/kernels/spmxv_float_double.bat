nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_FLOAT_DOUBLE -o ..\cubin\spmxv_float_double.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_float_double.cubin > ..\isa\spmxv_float_double.isa

