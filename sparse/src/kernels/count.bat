nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20  -o ../cubin/matrixcount.cubin matrixcount.cu
cuobjdump -sass ../cubin/matrixcount.cubin > ../isa/matrixcount.isa