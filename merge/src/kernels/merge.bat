nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/merge.cubin merge2.cu
cuobjdump -sass ../cubin/merge.cubin > ../isa/merge.isa
