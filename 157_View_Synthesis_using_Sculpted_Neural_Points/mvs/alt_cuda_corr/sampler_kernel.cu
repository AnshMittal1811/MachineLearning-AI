#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


#define BLOCK 16

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void sampler_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume.size(1);
  const int w1 = volume.size(2);
  const int h2 = volume.size(3);
  const int w2 = volume.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  const float x0 = coords[n][0][y][x];
  const float y0 = coords[n][1][y][x];

  const float dx = static_cast<scalar_t>(1.0 - (x0 - floor(x0)));
  const float dy = static_cast<scalar_t>(1.0 - (y0 - floor(y0)));

  int rd = 2*r + 1;
  for (int i=0; i<rd+1; i++) {
    for (int j=0; j<rd+1; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      if (within_bounds(y1, x1, h2, w2)) {
        scalar_t s = volume[n][y][x][y1][x1];

        if (i > 0 && j > 0)
          corr[n][i-1][j-1][y][x] += s * dx * dy;

        if (i > 0 && j < rd)
          corr[n][i-1][j][y][x] += s * dx * (1.0-dy);

        if (i < rd && j > 0)
          corr[n][i][j-1][y][x] += s * (1.0-dx) * dy;

        if (i < rd && j < rd)
          corr[n][i][j][y][x] += s * (1.0-dx) * (1.0-dy);

      }
    }
  }
}


template <typename scalar_t>
__global__ void sampler_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume_grad,
    int r)
{
      // batch index
  const int n = blockIdx.z;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int h1 = volume_grad.size(1);
  const int w1 = volume_grad.size(2);
  const int h2 = volume_grad.size(3);
  const int w2 = volume_grad.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  const float x0 = coords[n][0][y][x];
  const float y0 = coords[n][1][y][x];

  const float dx = x0 - floor(x0);
  const float dy = y0 - floor(y0);

  int rd = 2 * r + 1;
  for (int i=0; i<rd; i++) {
    for (int j=0; j<=rd; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      scalar_t g = 0.0;
      const scalar_t* ptr = &corr_grad[n][0][y][x];

      if (i>0 && j>0)
        g += *(ptr + rd*(j-1) + (i-1)) * (dy) * (dx);

      if (i>0 && n<rd)
        g += *(ptr + rd*j + (i-1)) * (1-dy) * (dx);

      if (i<rd && j>0)
        g += *(ptr + rd*(j-1) + i) * (dy) * (1-dx);

      if (i<rd && j<rd)
        g += *(ptr + rd*j + i) * (dy) * (1-dx);
        
      if (within_bounds(y1, x1, h2, w2)) {
        volume_grad[n][y][x][y1][x1] += g;
      }
    }
  }
}

std::vector<torch::Tensor> sampler_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK, BLOCK);

  auto opts = volume.options();
  torch::Tensor corr = torch::zeros(
    {batch_size, 2*radius+1, 2*radius+1, ht, wd}, opts);

  sampler_forward_kernel<float><<<blocks, threads>>>(
    volume.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    corr.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

/*
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
    sampler_forward_kernel1<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));
*/

  return {corr};

}


torch::Tensor sampler_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  auto volume_grad = torch::zeros_like(volume);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);

  AT_DISPATCH_FLOATING_TYPES(volume.type(), "sampler_backward_kernel", ([&] {
    sampler_backward_kernel<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      volume_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
  }));
  
  return volume_grad;
}

