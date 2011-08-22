nvcc -cubin -arch=compute_20 -code=sm_20 -Xptxas=-v -o ..\ptx\count.cubin countgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\count.cubin > ..\isa\count.isa

