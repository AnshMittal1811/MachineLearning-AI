#pragma once
#include <torch/extension.h>

void integrate(
    const torch::Tensor& rgb_sigma_tensor,
    const torch::Tensor& rgb_map_tensor,
    const torch::Tensor& acc_map_tensor,
    const torch::Tensor& transmittance_tensor,
    const float& transmittance_threshold,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const int& num_rays,
    const int& num_blocks,
    const int& num_threads
);

// It will skip empty index
void integrate_filter(
    const torch::Tensor& rgb_sigma_tensor,
    const torch::Tensor& rgb_map_tensor,
    const torch::Tensor& acc_map_tensor,
    const torch::Tensor& transmittance_tensor,
    const float& transmittance_threshold,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const torch::Tensor& rays_idx_tensor,
    const int& num_rays,
    const int& num_blocks,
    const int& num_threads
);

void replace_transparency_by_background_color(
    const torch::Tensor& rgb_map_tensor,
    const torch::Tensor& acc_map_tensor,
    const torch::Tensor& background_color_tensor,
    const int& num_blocks,
    const int& num_threads
);

void early_ray_filtering(
    const torch::Tensor& alphas_tensor,
    const torch::Tensor& output_masks_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const float& transmittance_threshold,
    const int& num_rays,
    const int& num_blocks,
    const int& num_threads
);