nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/select.cubin selectgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/select.cubin > ../isa/select.isa
