#pragma once
#include <torch/extension.h>

void sort_by_key_int16_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);
void sort_by_key_int32_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);
void sort_by_key_int64_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);
void sort_by_key_float32_int64(const torch::Tensor& keys_tensor, const torch::Tensor& values_tensor);