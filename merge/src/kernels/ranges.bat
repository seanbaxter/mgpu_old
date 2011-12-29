nvcc --cubin -Xptxas=-v -arch=compute_20 -code=sm_20  -o ../cubin/ranges.cubin ranges.cu
cuobjdump -sass ../cubin/ranges.cubin > ../isa/ranges.isa