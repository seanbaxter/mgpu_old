nvcc -m=32 --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ..\cubin\sorthist.cubin sorthist.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\sorthist.cubin > ..\isa\sorthist.isa

