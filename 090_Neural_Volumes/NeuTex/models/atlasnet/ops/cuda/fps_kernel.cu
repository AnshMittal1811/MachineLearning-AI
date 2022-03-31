/* CUDA Implementation for farthest point sampling*/
#ifndef _FPS_KERNEL
#define _FPS_KERNEL

#include <cmath>

#include <ATen/ATen.h>
#include <THC/THC.h>

// Note: AT_ASSERT has become AT_CHECK on master after 0.4.
// Note: AT_CHECK has become TORCH_CHECK on master after 1.2.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// Note: CHECK_EQ, CHECK_GT, etc. are marcos in Pytorch.
// #define CHECK_EQ(x, y) TORCH_CHECK(x == y, #x " does not equal to " #y)
// #define CHECK_GT(x, y) TORCH_CHECK(x > y, #x " is not greater than " #y)

#define MAX_THREADS 512

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}

#define RUN(BLOCK_SIZE, DIM) \
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "FarthestPointSample", ([&] { \
    FarthestPointSampleKernel<BLOCK_SIZE, DIM, scalar_t, int64_t> \
      <<<batch_size, BLOCK_SIZE>>>( \
      index.data<int64_t>(), \
      points.data<scalar_t>(), \
      temp.data<scalar_t>(), \
      num_points, \
      num_centroids); \
  }));

#define RUN_DIM(BLOCK_SIZE) \
  switch (dim) { \
    case 3: \
      RUN(BLOCK_SIZE, 3) \
      break; \
    case 2: \
      RUN(BLOCK_SIZE, 2) \
      break; \
    default: \
      TORCH_CHECK(false, "Only support dim=2 or 3."); \
  }

#define RUN_BLOCK(BLOCK_SIZE) \
  case BLOCK_SIZE: \
    RUN_DIM(BLOCK_SIZE) \
    break;

/*
Forward kernel
points: (B, N1, D)
temp: (B, N1)
index: (B, N2)
*/
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void FarthestPointSampleKernel(
    index_t* __restrict__ index,
    const scalar_t* __restrict__ points,
    scalar_t* __restrict__ temp,
    const int64_t num_points,
    const int64_t num_centroids) {
  // Allocate shared memory
  __shared__ scalar_t smem_dist[BLOCK_SIZE];
  // Use int to save memory
  __shared__ int smem_idx[BLOCK_SIZE];

  const int batch_idx = blockIdx.x;
  int cur_idx = 0;
  int points_offset = batch_idx * num_points * DIM;
  int temp_offset = batch_idx * num_points;
  int index_offset = batch_idx * num_centroids;

  // Explicitly choose the first point as a centroid
  if (threadIdx.x == 0) index[index_offset] = cur_idx;
  
  for (int i = 1; i < num_centroids; ++i) {
    scalar_t max_dist = 0.0;
    int max_idx = cur_idx;
    
    int offset1 = cur_idx * DIM;
    scalar_t coords1[DIM] = {0.0};
    #pragma unroll
    for (int ii = 0; ii < DIM; ++ii) {
      coords1[ii] = points[points_offset + offset1 + ii];
    }

    for (int j = threadIdx.x; j < num_points; j += BLOCK_SIZE) {
      int offset2 = j * DIM;
      scalar_t dist = 0.0;
      #pragma unroll
      for (int jj = 0; jj < DIM; ++jj) {
        scalar_t diff = points[points_offset + offset2 + jj] - coords1[jj];
        dist += diff * diff;
      }
      
      scalar_t last_dist = temp[temp_offset + j];
      if (last_dist > dist || last_dist < 0.0) {
        temp[temp_offset + j] = dist;
      } else {
        dist = last_dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
        max_idx = j;
      }
    }

    smem_dist[threadIdx.x] = max_dist;
    smem_idx[threadIdx.x] = max_idx;

    // assert block_size == blockDim.x
    int offset = BLOCK_SIZE / 2;
    while (offset > 0) {
      __syncthreads();
      if (threadIdx.x < offset) {
        scalar_t dist1 =  smem_dist[threadIdx.x];
        scalar_t dist2 = smem_dist[threadIdx.x+offset];
        if (dist1 < dist2) {
          smem_dist[threadIdx.x] = dist2;
          smem_idx[threadIdx.x] = smem_idx[threadIdx.x+offset];
        }
      }
      offset /= 2;
    }
    __syncthreads();

    cur_idx = smem_idx[0];
    if (threadIdx.x == 0) index[index_offset + i] = (index_t)cur_idx;
  }
}

/*
Forward interface
Input:
  points: (B, N1, D)
Output:
  index: (B, N2)
*/
at::Tensor FarthestPointSample(
	  const at::Tensor points,
    const int64_t num_centroids) {

	const auto batch_size = points.size(0);
	const auto num_points = points.size(1);
  const auto dim = points.size(2);

	// Sanity check
  CHECK_INPUT(points);
  TORCH_CHECK(dim == 2 || dim == 3, "Only support dim=2 or dim=3")
  CHECK_GT(num_centroids, 0);
  CHECK_GE(num_points, num_centroids);
	
  auto index = at::zeros({batch_size, num_centroids}, points.type().toScalarType(at::kLong));
  // In original implementation, it only allocates memory with the size of grid instead of batch size.
  auto temp = at::neg(at::ones({batch_size, num_points}, points.type()));

  // In order to make full use of shared memory and threads,
  // it is recommended to set num_centroids to be power of 2.
  const auto n_threads = opt_n_threads(num_points);
  
  switch (n_threads) {
    RUN_BLOCK(512)
    RUN_BLOCK(256)
    RUN_BLOCK(128)
    RUN_BLOCK(64)
    RUN_BLOCK(32)
    RUN_BLOCK(16)
    default:
      RUN_DIM(16)
  }

  THCudaCheck(cudaGetLastError());
  
  return index;
}

#endif