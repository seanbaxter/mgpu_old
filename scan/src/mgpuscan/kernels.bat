nvcc --cubin -Xptxas=-v -code=sm_20 -arch=compute_20 -o mgpuscan.cubin scangen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass mgpuscan.cubin > mgpuscan.isa 