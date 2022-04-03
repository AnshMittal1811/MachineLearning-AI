/* CUDA Implementation for feature interpolation
Pytorch has different APIs for different versions.
v1.1.0: type() -> scalar_type() for dispatch
v1.2.0: AT_CHECK -> TORCH_CHECK
*/
#ifndef _NN_DISTANCE_KERNEL
#define _NN_DISTANCE_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid
#include <THC/THC.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/********************************
* Forward kernel
*********************************/
template<typename scalar_t, typename index_t, uint64_t BLOCK_SIZE, uint64_t DIM>
__global__ void NNDistanceForwardKernel(
    const scalar_t *__restrict__ xyz1,
    const scalar_t *__restrict__ xyz2,
    scalar_t *__restrict__ dist,
    index_t *__restrict__ idx,
    const int64_t batch_size,
    const int64_t n1,
    const int64_t n2){
  // calculate the number of blocks
  const int64_t num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t num_block2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t total_blocks = batch_size * num_block1;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
    __shared__ scalar_t xyz2_buffer[BLOCK_SIZE * DIM];
    const int batch_idx = block_idx / num_block1;
    const int block_idx1 = block_idx % num_block1;
    const int xyz1_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int xyz1_offset = (batch_idx * n1 + xyz1_idx) * DIM;
    scalar_t coords1[DIM] = {0.0};
    if (xyz1_idx < n1) {
      #pragma unroll
      for (int i = 0; i < DIM; ++i) {
        coords1[i] = xyz1[xyz1_offset + i];
      }
    } else {
      // x1 = y1 = z1 = 0.0;
      ;
    }
    scalar_t min_dist = 1e32;
    index_t min_idx = -1;
    // load a block of xyz2 data to reduce the times to read data
    for (int block_idx2 = 0; block_idx2 < num_block2; ++block_idx2) {
      // load xyz2 data
      int xyz2_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int xyz2_offset = (batch_idx * n2 + xyz2_idx) * DIM;
      if (xyz2_idx < n2) {
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
          xyz2_buffer[threadIdx.x * DIM + i] = xyz2[xyz2_offset + i];
        }
      }
      __syncthreads();
      // calculate the distance between xyz1 and xyz2, with the shared memory.
      for (int j = 0; j < BLOCK_SIZE; ++j) {
        xyz2_idx = (block_idx2 * BLOCK_SIZE) + j;
        const int buffer_offset = j * DIM;
        scalar_t d = 0.0;
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
          scalar_t coord = xyz2_buffer[buffer_offset + i];
          d += (coord - coords1[i]) * (coord - coords1[i]);
        }
        if (xyz2_idx < n2 && d < min_dist) {
          min_dist = d;
          min_idx = xyz2_idx;
        }
      }
      __syncthreads();
    }
    if (xyz1_idx < n1) {
      const int output_offset = batch_idx * n1 + xyz1_idx;
      dist[output_offset] = min_dist;
      idx[output_offset] = min_idx;
    }
  }
}

inline bool getGrid(uint64_t numBlocks, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

#define CASE_RUN(DIM) \
  case DIM: \
    AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "NNDistanceForward", ([&] { \
      NNDistanceForwardKernel<scalar_t, int64_t, BLOCK_SIZE, DIM> \
        <<<grid1, BLOCK_SIZE>>>( \
          xyz1.data<scalar_t>(), \
          xyz2.data<scalar_t>(), \
          dist1.data<scalar_t>(), \
          idx1.data<int64_t>(), \
          batch_size, n1, n2); \
      })); \
    break;

/* Forward interface
Input:
  xyz1: (B, N1, 3)
  xyz2: (B, N2, 3)
Output:
  dist1: (B, N1)
  idx1: (B, N1)
  dist2: (B, N2)
  idx2: (B, N2)
*/
std::vector<at::Tensor> NNDistanceForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2){
  const auto batch_size = xyz1.size(0);
  const auto n1 = xyz1.size(1);
  const auto n2 = xyz2.size(1);
  const auto dim = xyz1.size(2);

  CHECK_EQ(xyz2.size(0), batch_size);
  CHECK_EQ(xyz2.size(2), dim);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);

  auto dist1 = at::zeros({batch_size, n1}, xyz1.type());
  auto idx1 = at::zeros({batch_size, n1}, xyz1.type().toScalarType(at::kLong));

  // Calculate grids and blocks for kernels
  const uint64_t BLOCK_SIZE = 512;
  const auto num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  dim3 grid1;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * num_block1, grid1, curDevice);

  // printf("(b, nb, n1, n2): (%ld, %ld, %ld, %ld)\n", batch_size, num_block1, n1, n2);
  
  switch (dim) {
    CASE_RUN(2)
    CASE_RUN(3)
    default:
      TORCH_CHECK(false, "Only support dim=2 or 3.");
  }
  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>({dist1, idx1});
}


/**********************************
* Backward kernel
***********************************/
/* Backward Kernel */
template <typename scalar_t, typename index_t, uint64_t DIM>
__global__ void NNDistanceBackwardKernel(
    const scalar_t *__restrict__ grad_dist,
    const index_t *__restrict__ index,
    const scalar_t *__restrict__ xyz1,
    const scalar_t *__restrict__ xyz2,
    scalar_t *__restrict__ grad_xyz1,
    scalar_t *__restrict__ grad_xyz2,
    const int64_t batch_size,
    const int64_t n1,
    const int64_t n2) {
  const uint64_t totalElements = batch_size * n1;
  for (int linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    int batch_idx  = linearId / n1;
    int xyz1_offset = linearId * DIM;
    int xyz2_offset = (batch_idx * n2 + index[linearId])* DIM;
    scalar_t g = grad_dist[linearId] * 2;
    scalar_t coord1, coord2, g_tmp;
    #pragma unroll
    for (int i = 0; i < DIM; ++i) {
      coord1 = xyz1[xyz1_offset + i];
      coord2 = xyz2[xyz2_offset + i];
      g_tmp = g * (coord1 - coord2);
      atomicAdd(grad_xyz1 + xyz1_offset + i, g_tmp);
      atomicAdd(grad_xyz2 + xyz2_offset + i, -g_tmp);
    }
  }
}

#define CASE_RUN2(DIM) \
  case DIM: \
    AT_DISPATCH_FLOATING_TYPES(grad_dist1.scalar_type(), "NNDistanceBackward", ([&] { \
      NNDistanceBackwardKernel<scalar_t, int64_t, DIM> \
        <<<grid1, block>>>( \
          grad_dist1.data<scalar_t>(), \
          idx1.data<int64_t>(), \
          xyz1.data<scalar_t>(), \
          xyz2.data<scalar_t>(), \
          grad_xyz1.data<scalar_t>(), \
          grad_xyz2.data<scalar_t>(), \
          batch_size, n1, n2); \
    })); \
    break;

/* Backward interface
Input:
  grad_dist1: (B, N1)
  xyz1: (B, N1, 3)
  xyz2: (B, N2, 3)
  idx1: (B, N1)
Output:
  grad_xyz1: (B, N1, 3)
  grad_xyz2: (B, N2, 3)
*/
std::vector<at::Tensor> NNDistanceBackward(
    const at::Tensor grad_dist1,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor idx1){
  const auto batch_size = grad_dist1.size(0);
  const auto n1 = xyz1.size(1);
  const auto n2 = xyz2.size(1);
  const auto dim = xyz1.size(2);
  CHECK_EQ(xyz1.size(0), batch_size);
  CHECK_EQ(xyz2.size(0), batch_size);
  CHECK_EQ(grad_dist1.size(1), n1);
  CHECK_EQ(xyz2.size(2), dim);
  CHECK_EQ(idx1.size(0), batch_size);
  CHECK_EQ(idx1.size(1), n1);
  CHECK_INPUT(grad_dist1);
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  CHECK_INPUT(idx1);

  auto grad_xyz1 = at::zeros({batch_size, n1, dim}, xyz1.type());
  auto grad_xyz2 = at::zeros({batch_size, n2, dim}, xyz2.type());
  // Calculate grids and blocks for kernels
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid1;
  const auto curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(batch_size * n1, grid1, curDevice), 1, "Too many elements to calculate");

  switch (dim) {
    CASE_RUN2(2)
    CASE_RUN2(3)
    default:
      TORCH_CHECK(false, "Only support dim=2 or 3.");
  }
  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>({grad_xyz1, grad_xyz2});
}

#endif