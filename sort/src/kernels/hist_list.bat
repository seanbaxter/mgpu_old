nvcc --cubin -Xptxas=-v -D INCLUDE_TRANSACTION_LIST -arch=compute_20 -code=sm_20 -o ..\cubin\hist_list.cubin histgen.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ..\cubin\hist_list.cubin > ..\isa\hist_list.isa

