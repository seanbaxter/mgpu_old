nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_FLOAT -o ..\cubin\spmxv_float.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_float.cubin > ..\isa\spmxv_float.isa

