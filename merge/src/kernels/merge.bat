nvcc --cubin -m=32 --open64 -Xptxas=-v -arch=compute_20 -code=sm_20 -o ../cubin/merge.cubin merge3.cu
cuobjdump -sass ../cubin/merge.cubin > ../isa/merge.isa
