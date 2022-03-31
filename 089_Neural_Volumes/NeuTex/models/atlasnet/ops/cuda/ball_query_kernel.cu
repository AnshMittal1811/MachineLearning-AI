/* CUDA Implementation for ball query*/
#ifndef _BALL_QUERY_KERNEL
#define _BALL_QUERY_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid
#include <THC/THC.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// NOTE: AT_CHECK has become TORCH_CHECK on master after 1.2.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline bool getGrid(uint64_t numBlocks, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

template <typename scalar_t, typename index_t, uint64_t BLOCK_SIZE>
__global__ void BallQueryForwardKernel(
    scalar_t* __restrict__ distance,
    index_t* __restrict__ count,
    const scalar_t *__restrict__ xyz1,
    const int64_t n1,
    const scalar_t *__restrict__ xyz2,
    const int64_t n2,
    const int64_t batch_size,
    const scalar_t radius) {

  // calculate the number of blocks
  const int64_t num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t num_block2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t total_blocks = batch_size * num_block1;
  const scalar_t radius_square = radius * radius;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
    __shared__ scalar_t xyz2_buffer[BLOCK_SIZE * 3];
    const int batch_idx = block_idx / num_block1;
    const int block_idx1 = block_idx % num_block1;
    const int xyz1_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int xyz1_offset = (batch_idx * n1 + xyz1_idx) * 3;
    scalar_t x1, y1, z1;
    if (xyz1_idx < n1) {
      x1 = xyz1[xyz1_offset + 0];
      y1 = xyz1[xyz1_offset + 1];
      z1 = xyz1[xyz1_offset + 2];
    } else {
      x1 = y1 = z1 = 0.0;
    }
    scalar_t dist_sum = 0.0;
    index_t neighbor_cnt = 0;
    // load a block of xyz2 data to reduce the times to read data
    for (int block_idx2 = 0; block_idx2 < num_block2; ++block_idx2) {
      // load xyz2 data
      int xyz2_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int xyz2_offset = (batch_idx * n2 + xyz2_idx) * 3;
      if (xyz2_idx < n2) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          xyz2_buffer[threadIdx.x * 3 + i] = xyz2[xyz2_offset + i];
        }
      }
      __syncthreads();
      // calculate the distance between xyz1 and xyz2, with the shared memory.
      for (int j = 0; j < BLOCK_SIZE; ++j) {
        xyz2_idx = (block_idx2 * BLOCK_SIZE) + j;
        const int buffer_offset = j * 3;
        scalar_t x2 = xyz2_buffer[buffer_offset + 0];
        scalar_t y2 = xyz2_buffer[buffer_offset + 1];
        scalar_t z2 = xyz2_buffer[buffer_offset + 2];
        scalar_t d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        // Note that we ignore too close points, which are regarded as points themselves.
        if (xyz2_idx < n2 && d < radius_square && d > 1e-16) {
          dist_sum += sqrt(d);
          neighbor_cnt += 1;
        }
      }
      __syncthreads();
    }
    if (xyz1_idx < n1) {
      const int output_offset = batch_idx * n1 + xyz1_idx;
      distance[output_offset] = dist_sum / scalar_t(max(neighbor_cnt, index_t(1)));
      count[output_offset] = neighbor_cnt;
    }
  }
}

/* Forward interface
Input:
  xyz1: (B, N1, 3)
  xyz2: (B, N2, 3)
  radius: scalar
Output:
  distance: (B, N1)
  count: (B, N1)
*/
std::vector<at::Tensor> BallQueryForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const float radius) {

  const auto batch_size = xyz1.size(0);
  const auto n1 = xyz1.size(1);
  const auto n2 = xyz2.size(1);

  // Sanity check
  CHECK_INPUT(xyz1);
  CHECK_INPUT(xyz2);
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);

  // Allocate new space for output
  auto distance = at::zeros({batch_size, n1}, xyz1.type());
  auto count = at::zeros({batch_size, n1}, xyz1.type().toScalarType(at::kLong));
  
  // Calculate grids and blocks for kernels
  const uint64_t BLOCK_SIZE = 512;
  const auto num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * num_block1, grid, curDevice);

  AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "BallQueryForward", ([&] {
    BallQueryForwardKernel<scalar_t, int64_t, BLOCK_SIZE>
      <<<grid, BLOCK_SIZE>>>(
      distance.data<scalar_t>(),
      count.data<int64_t>(),
      xyz1.data<scalar_t>(),
      n1,
      xyz2.data<scalar_t>(),
      n2,
      batch_size,
      (scalar_t)radius);
  }));

  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>({distance, count});
}

template <typename scalar_t, typename index_t, uint64_t BLOCK_SIZE>
__global__ void BallQueryBackwardKernel(
    scalar_t* __restrict__ grad_xyz1,
    scalar_t* __restrict__ grad_xyz2,
    const scalar_t* __restrict__ grad_dist,
    const index_t* __restrict__ count,
    const scalar_t *__restrict__ xyz1,
    const int64_t n1,
    const scalar_t *__restrict__ xyz2,
    const int64_t n2,
    const int64_t batch_size,
    const scalar_t radius) {

  // calculate the number of blocks
  const int64_t num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t num_block2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t total_blocks = batch_size * num_block1;
  const scalar_t radius_square = radius * radius;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
    __shared__ scalar_t xyz2_buffer[BLOCK_SIZE * 3];
    const int batch_idx = block_idx / num_block1;
    const int block_idx1 = block_idx % num_block1;
    const int xyz1_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int xyz1_offset = (batch_idx * n1 + xyz1_idx) * 3;
    scalar_t x1, y1, z1;
    if (xyz1_idx < n1) {
      x1 = xyz1[xyz1_offset + 0];
      y1 = xyz1[xyz1_offset + 1];
      z1 = xyz1[xyz1_offset + 2];
    } else {
      x1 = y1 = z1 = 0.0;
    }
    index_t neighbor_cnt = count[batch_idx * n1 + xyz1_idx];
    scalar_t g_all = grad_dist[batch_idx * n1 + xyz1_idx];
    // load a block of xyz2 data to reduce the times to read data
    for (int block_idx2 = 0; block_idx2 < num_block2; ++block_idx2) {
      // load xyz2 data
      int xyz2_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int xyz2_offset = (batch_idx * n2 + xyz2_idx) * 3;
      if (xyz2_idx < n2) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          xyz2_buffer[threadIdx.x * 3 + i] = xyz2[xyz2_offset + i];
        }
      }
      __syncthreads();
      // calculate the distance between xyz1 and xyz2, with the shared memory.
      for (int j = 0; j < BLOCK_SIZE; ++j) {
        xyz2_idx = (block_idx2 * BLOCK_SIZE) + j;
        const int buffer_offset = j * 3;
        scalar_t x2 = xyz2_buffer[buffer_offset + 0];
        scalar_t y2 = xyz2_buffer[buffer_offset + 1];
        scalar_t z2 = xyz2_buffer[buffer_offset + 2];
        scalar_t d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        // Note that we ignore too close points, which are regarded as points themselves.
        if (xyz1_idx < n1 && xyz2_idx < n2 && d < radius_square && d > 1e-16) {
          scalar_t g = g_all / (neighbor_cnt * sqrt(d));
          scalar_t gx = g * (x1 - x2);
          scalar_t gy = g * (y1 - y2);
          scalar_t gz = g * (z1 - z2);
          int xyz2_offset = (batch_idx * n2 + xyz2_idx) * 3;
          atomicAdd(grad_xyz1 + xyz1_offset + 0, gx);
          atomicAdd(grad_xyz1 + xyz1_offset + 1, gy);
          atomicAdd(grad_xyz1 + xyz1_offset + 2, gz);
          atomicAdd(grad_xyz2 + xyz2_offset + 0, -gx);
          atomicAdd(grad_xyz2 + xyz2_offset + 1, -gy);
          atomicAdd(grad_xyz2 + xyz2_offset + 2, -gz);
        }
      }
      __syncthreads();
    }
  }
}

/* Backward interface
Input:
  grad_dist: (B, N1)
  count: (B, N1)
  xyz1: (B, N1, 3)
  xyz2: (B, N2, 3)
  radius: scalar
Output:
  grad_xyz1: (B, N1, 3)
  grad_xyz2: (B, N2, 3)
*/
std::vector<at::Tensor> BallQueryBackward(
    const at::Tensor grad_dist,
    const at::Tensor count,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const float radius){
  const auto batch_size = grad_dist.size(0);
  const auto n1 = xyz1.size(1);
  const auto n2 = xyz2.size(1);
  CHECK_EQ(count.size(0), batch_size);
  CHECK_EQ(xyz1.size(0), batch_size);
  CHECK_EQ(xyz2.size(0), batch_size);
  CHECK_EQ(grad_dist.size(1), n1);
  CHECK_EQ(count.size(1), n1);
  CHECK_EQ(xyz1.size(2), 3);
  CHECK_EQ(xyz2.size(2), 3);

  auto grad_xyz1 = at::zeros({batch_size, n1, 3}, grad_dist.type());
  auto grad_xyz2 = at::zeros({batch_size, n2, 3}, grad_dist.type());
  // Calculate grids and blocks for kernels
  const uint64_t BLOCK_SIZE = 512;
  const auto num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * num_block1, grid, curDevice);

  AT_DISPATCH_FLOATING_TYPES(xyz1.scalar_type(), "BallQueryBackward", ([&] {
    BallQueryBackwardKernel<scalar_t, int64_t, BLOCK_SIZE>
      <<<grid, BLOCK_SIZE>>>(
      grad_xyz1.data<scalar_t>(),
      grad_xyz2.data<scalar_t>(),
      grad_dist.data<scalar_t>(),
      count.data<int64_t>(),
      xyz1.data<scalar_t>(),
      n1,
      xyz2.data<scalar_t>(),
      n2,
      batch_size,
      (scalar_t)radius);
  }));

  return std::vector<at::Tensor>({grad_xyz1, grad_xyz2});

}

#endif