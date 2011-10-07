nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/ksmallest.cubin ksmallest.cu
IF %ERRORLEVEL% EQU 0 cuobjdump -sass ../cubin/ksmallest.cubin > ../isa/ksmallest.isa
