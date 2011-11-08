nvcc --cubin -Xptxas=-v -D MAT_TYPE_CFLOAT -arch=compute_20 -code=sm_20 -o ../cubin/spmxv_cfloat.cubin spmxv.cu
cuobjdump -sass ../cubin/spmxv_cfloat.cubin > ../isa/spmxv_cfloat.isa
