nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -D USE_FLOAT -D USE_COMPLEX -o ..\cubin\spmxv_cfloat.cubin spmxvgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\spmxv_cfloat.cubin > ..\isa\spmxv_cfloat.isa

