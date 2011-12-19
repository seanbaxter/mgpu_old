nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/bwt.cubin bwt.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/bwt.cubin > ../isa/bwt.isa