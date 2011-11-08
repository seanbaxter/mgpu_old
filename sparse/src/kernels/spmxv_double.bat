nvcc --cubin -Xptxas=-v -D MAT_TYPE_DOUBLE -arch=compute_20 -code=sm_20 -o ../cubin/spmxv_double.cubin spmxv.cu
cuobjdump -sass ../cubin/spmxv_double.cubin > ../isa/spmxv_double.isa
