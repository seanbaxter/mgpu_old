nvcc -m=32 -cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ..\cubin\count.cubin countgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\count.cubin > ..\isa\count.isa

