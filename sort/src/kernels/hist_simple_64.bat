nvcc -m=64 --cubin -D BUILD_64 -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin64/hist_simple.cubin histgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin64/hist_simple.cubin > ../isa64/hist_simple.isa

