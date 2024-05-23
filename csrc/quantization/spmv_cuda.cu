#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "spmv_cuda.h"


__global__ void spmv_kernel(
  const  int*  rows, // row_idx
  const  int*  cols, // col_ide
  const  half*  mat, // val_arr
  const  half*  vec, // input x
         half*  mul, // output y
  const  int num_rows // number of rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {

        float dot = 0.0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        // if(row==0)
          // printf("Row:%0d, start:%0d, end:%0d \n",row, start_elem, end_elem);

        for (int i = start_elem; i < end_elem; i++) {
            float a = __half2float(mat[i]);
            float b = __half2float(vec[cols[i]]);
            float tmp = a * b;
            dot += tmp;
            // if(row==0)
            // printf("Row:%0d, start:%0d, end:%0d, a:%0f, b:%0f, tmp:%0f \n",row, start_elem, end_elem, a, b, tmp);
            // dot += mat[i] * vec[cols[i]];
        }
        // printf("KIMHO Row:%0d dot:%0f\n",row, dot);
        // atomicAdd(&mul[row], __float2half(dot));
        mul[row] = __float2half(dot);
    }
}


torch::Tensor spmv_forward_cuda(
    torch::Tensor _sp_in_feats,
    torch::Tensor _row_Ptr,
    torch::Tensor _col_Idx,
    torch::Tensor _spmat,
    int num_rows,
    int feature_size
    )
{

    auto options = torch::TensorOptions().dtype(_sp_in_feats.dtype()).device(_sp_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_rows, feature_size}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    auto rowPtr = reinterpret_cast<int*>(_row_Ptr.data_ptr<int>());
    auto colIdx = reinterpret_cast<int*>(_col_Idx.data_ptr<int>());
    auto spmat = reinterpret_cast<half*>(_spmat.data_ptr<at::Half>());
    auto sp_in_feats = reinterpret_cast<half*>(_sp_in_feats.data_ptr<at::Half>());

    dim3 spmv_block((num_rows + 32 - 1) / 32);
    dim3 spmv_threads(32);

    // printf("rowPtr:%0d %0d\n",rowPtr[0],rowPtr[5]);

    spmv_kernel<<<spmv_block, spmv_threads>>>(
      rowPtr, colIdx, 
      spmat,
      sp_in_feats,
      out_feats,
      num_rows
    );
    
    return _out_feats;
}