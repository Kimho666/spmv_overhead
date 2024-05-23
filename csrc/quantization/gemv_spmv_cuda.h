#pragma once
#include <torch/extension.h>

torch::Tensor gemv_spmv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size,
    torch::Tensor _sp_in_feats,
    torch::Tensor _row_Ptr,
    torch::Tensor _col_Idx,
    torch::Tensor _spmat,
    int num_rows,
    int feature_size);