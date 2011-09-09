nvcc -m=64 --cubin -D BUILD_64 -Xptxas=-v -D INCLUDE_TRANSACTION_LIST -arch=compute_20 -code=sm_20 -o ../cubin64/hist_list.cubin histgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin64/hist_list.cubin > ../isa64/hist_list.isa

