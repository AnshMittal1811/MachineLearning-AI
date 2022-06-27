#pragma once
#include <torch/extension.h>

torch::Tensor get_rays_d(
    const int& H,
    const int& W,
    const float& px,
    const float& py,
    const float& fx,
    const float& fy,
    const torch::Tensor& c2w_tensor,
    const torch::Tensor& ndc_points_tensor,
    const int& root_num_blocks,
    const int& root_num_threads
);

void compute_ray_plane_intersection(
    const torch::Tensor& planes_center_tensor,
    const torch::Tensor& planes_normal_tensor,
    const torch::Tensor& planes_x_tensor,
    const torch::Tensor& planes_y_tensor,
    const torch::Tensor& planes_w_tensor,
    const torch::Tensor& planes_h_tensor,
    const torch::Tensor& rays_tensor,
    const torch::Tensor& camera_center_tensor,
    const torch::Tensor& hits_tensor,
    const int& num_blocks,
    const int& num_threads
);

void compute_ray_plane_intersection_mt(
    const torch::Tensor& planes_vertices_tensor,
    const torch::Tensor& rays_tensor,
    const torch::Tensor& camera_center_tensor,
    const torch::Tensor& hits_tensor,
    const int& num_blocks,
    const int& num_threads
);

void store_ray_plane_intersection_mt(
    const torch::Tensor& planes_vertices_tensor,
    const torch::Tensor& rays_tensor,
    const torch::Tensor& rays_idx_tensor,
    const torch::Tensor& camera_center_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const torch::Tensor& points_tensor,
    const torch::Tensor& view_dirs_tensor,
    const torch::Tensor& depths_tensor,
    const int& num_blocks,
    const int& num_threads
);