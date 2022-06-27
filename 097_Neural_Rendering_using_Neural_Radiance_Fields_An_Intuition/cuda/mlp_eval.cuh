#pragma once
#include <torch/extension.h>

torch::Tensor mlp_eval_1d(
    const torch::Tensor& points_tensor,
    const torch::Tensor& viewdir_tensor,
    const torch::Tensor& model_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const int& num_networks,
    const int& num_threads
);

// It will skip empty index
torch::Tensor mlp_eval_1d_filter(
    const torch::Tensor& points_tensor,
    const torch::Tensor& viewdir_tensor,
    const torch::Tensor& model_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const torch::Tensor& planes_idx_tensor,
    const int& num_networks,
    const int& num_threads
);