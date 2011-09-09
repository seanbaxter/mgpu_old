nvcc -m=64 -cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin64/count.cubin countgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin64/count.cubin > ../isa64/count.isa

