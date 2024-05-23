# spmv_overhead
Profile the speed degradation of GEMV and dequant+GEMV with SpMV.

# How to profile it?
Now, we have integrated spmv with CSR format input with AWQ GEMV kernel with 4-bit dequantization. 
cd spmv_overhead
python setup.py install
python profile_gemv_spmv.py
