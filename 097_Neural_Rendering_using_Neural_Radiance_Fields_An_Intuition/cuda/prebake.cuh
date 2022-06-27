#pragma once
#include <torch/extension.h>

void sample_from_planes_alpha(
    const torch::Tensor& points_tensor,
    const torch::Tensor& output_alphas_tensor,
    const torch::Tensor& planes_params_tensor,
    const torch::Tensor& planes_alphas_tensor,
    const int& bake_res,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const int& num_blocks,
    const int& num_threads
);