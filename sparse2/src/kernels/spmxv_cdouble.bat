nvcc --cubin -Xptxas=-v -D MAT_TYPE_CDOUBLE -arch=compute_20 -code=sm_20 -o ../cubin/spmxv_cdouble.cubin spmxv.cu
cuobjdump -sass ../cubin/spmxv_cdouble.cubin > ../isa/spmxv_cdouble.isa
