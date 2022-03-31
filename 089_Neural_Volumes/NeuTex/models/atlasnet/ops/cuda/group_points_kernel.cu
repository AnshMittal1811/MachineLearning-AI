/* CUDA Implementation for efficient gather*/
#ifndef _GROUP_POINTS_KERNEL
#define _GROUP_POINTS_KERNEL

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

// NOTE: AT_ASSERT has become TORCH_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;

/* 
Forward interface
Input:
  input: (B, C, N1)
  index: (B, N2, K)
Output:
  output: (B, C, N2, K)
*/
at::Tensor GroupPointsForward(
    const at::Tensor input,
    const at::Tensor index) {
  const auto batch_size = input.size(0);
  const auto channels = input.size(1);
  const auto num_inst = input.size(2);
  const auto num_select = index.size(1);
  const auto k = index.size(2);

  // Sanity check
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(index.size(0), batch_size);

  auto input_expand = input.unsqueeze(2).expand({batch_size, channels, num_select, num_inst});  // (B, C, N2, N1)
  auto index_expand = index.unsqueeze(1).expand({batch_size, channels, num_select, k});  // (B, C, N2, K)

  auto output = input_expand.gather(3, index_expand);  // (B, C, N2, K)

  return output;
}

/* Backward Kernel */
template <typename scalar_t, typename index_t>
__global__ void GroupPointsBackwardKernel(
    const TensorInfo<scalar_t, index_t> grad_input,
    const TensorInfo<scalar_t, index_t> grad_output,
    const TensorInfo<int64_t, index_t> index,
    const index_t totalElements) {
  index_t channels = grad_input.sizes[1];
  index_t num_inst = grad_input.sizes[2];
  index_t num_select = index.sizes[1];
  index_t k = index.sizes[2];
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    // Compute offsets
    index_t linearId_tmp = linearId;
    index_t k_offset = linearId_tmp % k;
    linearId_tmp /= k;
    index_t inst_offset = linearId_tmp % num_select;
    linearId_tmp /= num_select;
    index_t channel_offset = linearId_tmp % channels;
    index_t batch_offset = linearId_tmp / channels;
    
    index_t srcOffset = k_offset * grad_output.strides[3]
      + inst_offset * grad_output.strides[2]
      + channel_offset * grad_output.strides[1]
      + batch_offset * grad_output.strides[0];

    index_t tensorOffset = channel_offset * grad_input.strides[1]
      + batch_offset * grad_input.strides[0];
    
    index_t indexOffset = k_offset * index.strides[2]
      + inst_offset * index.strides[1]
      + batch_offset * index.strides[0];

    int64_t indexValue = index.data[indexOffset];
    assert(indexValue >= 0 && indexValue < num_inst);
    tensorOffset += indexValue * grad_input.strides[2];
    atomicAdd(&grad_input.data[tensorOffset], grad_output.data[srcOffset]);
  }
}

/* 
Backward interface
Input:
  grad_output: (B, C, N2, K)
  index: (B, N2, K)
Output:
  grad_input: (B, C, N1)
*/
at::Tensor GroupPointsBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const int64_t num_points) {
  const auto batch_size = grad_output.size(0);
  const auto channels = grad_output.size(1);
  const auto num_select = grad_output.size(2);
  const auto k = grad_output.size(3);

  // Sanity check
  CHECK_CUDA(grad_output);
  CHECK_CUDA(index);
  CHECK_EQ(grad_output.dim(), 4);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(index.size(0), batch_size);
  CHECK_EQ(index.size(1), num_select);
  CHECK_EQ(index.size(2), k);

  // Allocate new space for output
  auto grad_input = at::zeros({batch_size, channels, num_points}, grad_output.type());
  CHECK_CUDA(grad_input);
  CHECK_CONTIGUOUS(grad_input);

  // Calculate grids and blocks for kernels 
  const auto totalElements = grad_output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "GroupPointsBackward", ([&] {
    auto gradInputInfo = getTensorInfo<scalar_t, uint64_t>(grad_input);
    auto gradOutputInfo = getTensorInfo<scalar_t, uint64_t>(grad_output);
    auto IndexInfo = getTensorInfo<int64_t, uint64_t>(index);
    GroupPointsBackwardKernel<scalar_t, uint64_t>
      <<<grid, block>>>(
        gradInputInfo,
        gradOutputInfo,
        IndexInfo,
        (uint64_t)totalElements);
  }));

  THCudaCheck(cudaGetLastError());

  return grad_input;
}
#endif