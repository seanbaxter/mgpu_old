nvcc --cubin -Xptxas=-v -D MAT_TYPE_FLOAT -arch=compute_20 -code=sm_20 -o ../cubin/spmxv_float.cubin spmxv.cu
cuobjdump -sass ../cubin/spmxv_float.cubin > ../isa/spmxv_float.isa
