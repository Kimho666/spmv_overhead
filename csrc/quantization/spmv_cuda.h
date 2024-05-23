#pragma once
#include <torch/extension.h>

torch::Tensor spmv_forward_cuda(
    torch::Tensor _sp_in_feats,
    torch::Tensor _row_Ptr,
    torch::Tensor _col_Idx,
    torch::Tensor _spmat,
    int num_rows,
    int feature_size
    );