#include "cuda.h"
#include "stdio.h"
#include "prebake.cuh"
#include "utils.cuh"


__global__ void sample_from_planes_alpha_kernel(
    const float *__restrict__ points,
    float *__restrict__ output_alphas,
    const float *__restrict__ planes_params,
    const float *__restrict__ planes_alphas,
    const int bake_res,
    const int *__restrict__ starts,
    const int *__restrict__ ends
)
{
    // planes_x_basis, planes_y_basis, planes_center, planes_w, planes_h (3 + 3 + 3 + 1 + 1 = 11)
    constexpr int params_dim = 11;
    constexpr int pos_dim = 3;

    // load parameters for specific plane
    float plane_x_basis[3], plane_y_basis[3], plane_center[3], plane_w, plane_h;
    int params_offset = blockIdx.x * params_dim;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        plane_x_basis[i] = planes_params[params_offset + i];
        plane_y_basis[i] = planes_params[params_offset + 3 + i];
        plane_center[i] = planes_params[params_offset + 6 + i];
    }
    plane_w = planes_params[params_offset + 9];
    plane_h = planes_params[params_offset + 10];

    int alpha_offset = blockIdx.x * bake_res * bake_res;    // base offset of planes alpha matrix

    int idx = starts[blockIdx.x] + threadIdx.x;
    while (idx < ends[blockIdx.x]) {
        
        float position_world[3];  // point in world coordinate
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            position_world[i] = points[idx * pos_dim + i];
        }

        // project to plane space
        float position_plane[2] = {0, 0};  // point in plane coordinate
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            float offset = position_world[i] - plane_center[i];
            position_plane[0] += offset * plane_x_basis[i];
            position_plane[1] += offset * plane_y_basis[i];
        }

        // convert: plane space (-w/2 ~ w/2) -> normalized plane space (-1 ~ 1) -> grid space (0 ~ bake_res)
        float norm_x = position_plane[0] / plane_w * -2;
        float norm_y = position_plane[1] / plane_h * -2;
        // float grid_x = (norm_x + 1) / 2 * (bake_res - 1);  // align_corners = True ([0, size - 1])
        // float grid_y = (norm_y + 1) / 2 * (bake_res - 1);
        float grid_x = ((norm_x + 1) * bake_res - 1) / 2;     // align_corners = False ([-0.5, size - 0.5]) -> pytorch3d convention
        float grid_y = ((norm_y + 1) * bake_res - 1) / 2;

        // query with bilinear interpolation
        float output_alpha = 0.0;
        if (grid_x >= 0 && grid_x < bake_res - 1 && grid_y >= 0 && grid_y < bake_res - 1) {

            int i = floor(grid_x);
            int j = floor(grid_y);
            float a = grid_x - i;
            float b = grid_y - j;

            float h00 = planes_alphas[alpha_offset + j * bake_res + i];
            float h10 = planes_alphas[alpha_offset + j * bake_res + (i + 1)];
            float h01 = planes_alphas[alpha_offset + (j + 1) * bake_res + i];
            float h11 = planes_alphas[alpha_offset + (j + 1) * bake_res + (i + 1)];

            float tmp1 = (1 - a) * h00 + a * h10;
            float tmp2 = (1 - a) * h01 + a * h11;
            output_alpha = (1 - b) * tmp1 + b * tmp2;
        }

        output_alphas[idx] = output_alpha;

        idx += blockDim.x;
    }
}


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
)
{
    float *points = points_tensor.data_ptr<float>();
    float *output_alphas = output_alphas_tensor.data_ptr<float>();
    float *planes_params = planes_params_tensor.data_ptr<float>();
    float *planes_alphas = planes_alphas_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();

    sample_from_planes_alpha_kernel<<<num_blocks, num_threads, 0>>>(points, output_alphas, planes_params, planes_alphas, bake_res, starts, ends);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}