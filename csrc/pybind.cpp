#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quantization/gemv_cuda.h"
// #include "quantization/gemv_mxq_cuda.h"
// #include "quantization/spmv_cuda.h"
#include "quantization/gemv_spmv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    // m.def("gemv_mxq_forward_cuda", &gemv_mxq_forward_cuda, "Quantized GEMV mxq kernel.");
    // m.def("spmv_forward_cuda", &spmv_forward_cuda, "Quantized SPMV kernel.");
    m.def("gemv_spmv_forward_cuda", &gemv_spmv_forward_cuda, "Quantized GEMV SPMV kernel.");
}
