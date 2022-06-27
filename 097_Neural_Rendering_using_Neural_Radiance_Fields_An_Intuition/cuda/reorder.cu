#include "reorder.cuh"
#include "utils.cuh"
#include <thrust/sort.h>
#include <thrust/gather.h>

void sort_by_key_int16_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    int16_t *keys = keys_tensor.data_ptr<int16_t>();
    int64_t *values = values_tensor.data_ptr<int64_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}

void sort_by_key_int32_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    int32_t *keys = keys_tensor.data_ptr<int32_t>();
    int64_t *values = values_tensor.data_ptr<int64_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}

void sort_by_key_int64_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    int64_t *keys = keys_tensor.data_ptr<int64_t>();
    int64_t *values = values_tensor.data_ptr<int64_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}

void sort_by_key_float32_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor)
{
    int length = keys_tensor.size(0);
    float *keys = keys_tensor.data_ptr<float>();
    int64_t *values = values_tensor.data_ptr<int64_t>();
    thrust::sort_by_key(thrust::device, keys, keys + length, values);
}
