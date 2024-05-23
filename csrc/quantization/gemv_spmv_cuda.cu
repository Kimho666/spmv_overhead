#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_spmv_cuda.h"
#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define PACK_FACTOR_2b 16
#define WARP_SIZE 32

// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128_for_spmv(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // consistent with input shape
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
    //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      float current_zeros = (float)((packed_zeros >> (threadIdx.x / 4 * 4)) & 0xF);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
    //  outputs[oc_idx] = __float2half(psum); 
      atomicAdd(&outputs[oc_idx], __float2half(psum));
    }
}


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
    int feature_size
    )
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<uint32_t*>(_zeros.data_ptr<int>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(0)}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    auto rowPtr = reinterpret_cast<int*>(_row_Ptr.data_ptr<int>());
    auto colIdx = reinterpret_cast<int*>(_col_Idx.data_ptr<int>());
    auto spmat = reinterpret_cast<half*>(_spmat.data_ptr<at::Half>());
    auto sp_in_feats = reinterpret_cast<half*>(_sp_in_feats.data_ptr<at::Half>());

    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);

    dim3 spmv_block((num_rows + 32 - 1) / 32);
    dim3 spmv_threads(32);

    spmv_kernel<<<spmv_block, spmv_threads>>>(
      rowPtr, colIdx, spmat,
      sp_in_feats,
      out_feats,
      num_rows
    );

    gemv_kernel_g128_for_spmv<<<num_blocks, num_threads>>>(
    // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
    // constants
      num_in_channels, num_out_channels
    );
    
    return _out_feats;
}