// CUDA Implementation for feature interpolation
#ifndef _KNN_DISTANCE_KERNEL
#define _KNN_DISTANCE_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_THREADS 512

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}

// From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
inline bool getGrid(uint64_t numBlocks, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

/****************************
* Kernel for searching point
*****************************/
template <unsigned int BLOCK_SIZE, unsigned int K, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void KNNDistanceKernel(
    index_t *__restrict__ index,
    scalar_t *__restrict__ distance,
    const scalar_t *__restrict__ query,
    const scalar_t *__restrict__ key,
    const int64_t batch_size,  
    const int64_t num_query,  
    const int64_t num_key){

  // calculate the number of blocks
  const int num_blocks1 = (num_query + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int num_blocks2 = (num_key + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int total_blocks = batch_size * num_blocks1;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    const int batch_idx = block_idx / num_blocks1;
    const int block_idx1 = block_idx % num_blocks1;
    const int query_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int query_offset = (batch_idx * num_query + query_idx) * DIM;

    // load current query point
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < num_query) {
      #pragma unroll
      for (int i = 0; i < DIM; ++i) {
        cur_query[i] = query[query_offset + i];
      }
    }

    // record topk
    scalar_t min_dist[K] = {1e40};
    int min_idx[K] = {-1};
    
    // load a block of key data to reduce the time to read data
    for (int block_idx2 = 0; block_idx2 < num_blocks2; ++block_idx2) {
      // load key data
      int key_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int key_offset = (batch_idx * num_key + key_idx) * DIM;
      if (key_idx < num_key) {
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
          key_buffer[threadIdx.x * DIM + i] = key[key_offset + i];
        }
      }
      __syncthreads();
      
      // calculate the distance between current query and key, with the shared memory.
      if (query_idx < num_query) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
          int key_idx2 = (block_idx2 * BLOCK_SIZE) + j;
          const int buffer_offset = j * DIM;
          scalar_t dist = 0.0;
          #pragma unroll
          for (int i = 0; i < DIM; ++i) {
            scalar_t diff = key_buffer[buffer_offset + i] - cur_query[i];
            dist += diff * diff;
          }
          if (key_idx2 < num_key) {
            // update min distance
            #pragma unroll
            for (int k = 0; k < K; ++k) {
              if (dist < min_dist[k]) {
                for (int l = K - 1; l > k; --l) {
                  min_dist[l] = min_dist[l - 1];
                  min_idx[l] = min_idx[l - 1];
                }
                min_dist[k] = dist;
                min_idx[k] = key_idx2;
                break;
              }
            }
          }
        }
      }
      __syncthreads();
    }

    // output
    const int out_offset = (batch_idx * num_query + query_idx) * K;
    if (query_idx < num_query) {
      #pragma unroll
      for (int k = 0; k < K; ++k) {
        index[out_offset + k] = min_idx[k];
        distance[out_offset + k] = min_dist[k];
      }
    }
  }
}

#define RUN(BLOCK_SIZE, K, DIM) \
  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "KNNDistance", ([&] { \
    KNNDistanceKernel<BLOCK_SIZE, K, DIM, scalar_t, int64_t> \
      <<<grid, BLOCK_SIZE>>>( \
      index.data<int64_t>(), \
      distance.data<scalar_t>(), \
      query.data<scalar_t>(), \
      key.data<scalar_t>(), \
      batch_size, \
      num_query, \
      num_key); \
    }));

#define RUN_DIM(BLOCK_SIZE, K, DIM) \
  switch (DIM) { \
    case 2: \
      RUN(BLOCK_SIZE, K, 2) \
      break; \
    case 3: \
      RUN(BLOCK_SIZE, K, 3) \
      break; \
    default: \
      TORCH_CHECK(false, "Only support dim=2 or 3."); \
  }

#define RUN_K(BLOCK_SIZE, K, DIM) \
  switch (K) { \
    case 3: \
      RUN_DIM(BLOCK_SIZE, 3, DIM) \
      break; \
    case 5: \
      RUN_DIM(BLOCK_SIZE, 5, DIM) \
      break; \
    case 10: \
      RUN_DIM(BLOCK_SIZE, 10, DIM) \
      break; \
    default: \
      TORCH_CHECK(false, "Only support k=3, 5, 10."); \
  }

#define RUN_CASE(BLOCK_SIZE) \
  case BLOCK_SIZE: \
    RUN_K(BLOCK_SIZE, k, dim) \
    break;

/* 
Forward interface
Input:
  query: (B, N1, 3)
  key: (B, N2, 3)
  k: int
Output:
  index: (B, N1, K)
  distance: (B, N1, K)
*/
std::vector<at::Tensor> KNNDistance(
    const at::Tensor query,
    const at::Tensor key,
    const int64_t k) {
  
  const auto batch_size = query.size(0);
  const auto num_query = query.size(1);
  const auto dim = query.size(2);
  const auto num_key = key.size(1);

  // sanity check
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_EQ(key.size(0), batch_size);
  // CHECK_EQ(dim, 3);
  CHECK_EQ(key.size(2), dim);
  CHECK_GE(num_key, k);
  // TORCH_CHECK(k == 3, "Only support 3-NN.");

  auto index = at::zeros({batch_size, num_query, k}, query.type().toScalarType(at::kLong));
  auto distance = at::zeros({batch_size, num_query, k}, query.type());

  // Calculate grids and blocks for kernels 
  const auto n_threads = opt_n_threads(min(num_query, num_key));
  const auto n_blocks = (num_query + n_threads - 1) / n_threads;
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * n_blocks, grid, curDevice);

  switch (n_threads) {
    RUN_CASE(512)
    RUN_CASE(256)
    RUN_CASE(128)
    RUN_CASE(64)
    RUN_CASE(32)
    default:
      RUN_K(16, k, dim)
  }
  
  THCudaCheck(cudaGetLastError());
  
  return std::vector<at::Tensor>({index, distance});
}

#endif
