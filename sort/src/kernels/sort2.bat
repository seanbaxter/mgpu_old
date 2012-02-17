nvcc --open64 -m=32 --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/sort2.cubin sort2.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sort2.cubin > ../isa/sort2.isa


