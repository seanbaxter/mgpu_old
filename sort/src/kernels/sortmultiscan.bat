nvcc -m=32 --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/sortmultiscan.cubin sortmultiscan.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/sortmultiscan.cubin > ../isa/sortmultiscan.isa


