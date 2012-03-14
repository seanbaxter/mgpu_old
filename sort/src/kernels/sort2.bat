nvcc --open64 -m=32 --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/sortcompact.cubin sortcompact.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sortcompact.cubin > ../isa/sortcompact.isa


