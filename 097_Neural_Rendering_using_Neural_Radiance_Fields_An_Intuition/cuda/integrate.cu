#include "integrate.cuh"
#include "utils.cuh"


__global__ void integrate_kernel(
    const float4 *__restrict__ rgb_sigma,
    float3 *__restrict__ rgb_map,
    float *__restrict__ acc_map,
    float *__restrict__ transmittance,
    const int *__restrict__ starts,
    const int *__restrict__ ends,
    const int num_rays
)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (ray_idx < num_rays) {
        
        float my_transmittance = 1.0f;
        float3 rgb_out = {0.0f, 0.0f, 0.0f};
        float acc_out = 0.0f;

        int sample_idx = starts[ray_idx];
        int sample_end_idx = ends[ray_idx];
        while (sample_idx < sample_end_idx) {
            float4 my_rgb_sigma = rgb_sigma[sample_idx];
            float alpha = my_rgb_sigma.w;
            float weight = alpha * my_transmittance;
            my_transmittance *= 1.0f - alpha + 1e-10;
            rgb_out.x += my_rgb_sigma.x * weight;
            rgb_out.y += my_rgb_sigma.y * weight;
            rgb_out.z += my_rgb_sigma.z * weight;
            acc_out += weight;
            sample_idx++;
        }

        transmittance[ray_idx] = my_transmittance;
        rgb_map[ray_idx] = rgb_out;
        acc_map[ray_idx] = acc_out;

        ray_idx += gridDim.x * blockDim.x;
    }
}


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
)
{
    float4 *rgb_sigma = (float4*)rgb_sigma_tensor.data_ptr<float>();
    float3 *rgb_map = (float3*)rgb_map_tensor.data_ptr<float>();
    float *acc_map = acc_map_tensor.data_ptr<float>();
    float *transmittance = transmittance_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();

    integrate_kernel<<<num_blocks, num_threads, 0>>>(rgb_sigma, rgb_map, acc_map, transmittance, starts, ends, num_rays);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


__device__ __constant__ float3 c_background_color;
__global__ void replace_transparency_by_background_color_kernel(
    float3 *__restrict__ rgb_map,
    const float *__restrict__ acc_map,
    const int num_pixels
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < num_pixels) {
        float3 in = rgb_map[idx];
        float t = 1.0f - acc_map[idx];
        float3 out;
        out.x = in.x + c_background_color.x * t;
        out.y = in.y + c_background_color.y * t;
        out.z = in.z + c_background_color.z * t;
        rgb_map[idx] = out;
        idx += gridDim.x * blockDim.x;
    }
}

void replace_transparency_by_background_color(
    const torch::Tensor& rgb_map_tensor,
    const torch::Tensor& acc_map_tensor,
    const torch::Tensor& background_color_tensor,
    const int& num_blocks,
    const int& num_threads
)
{
    float3 *rgb_map = (float3*)rgb_map_tensor.data_ptr<float>();
    float *acc_map = acc_map_tensor.data_ptr<float>();
    float3 *background_color = (float3*)background_color_tensor.data_ptr<float>();
    cudaMemcpyToSymbol(c_background_color, background_color, 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    int num_pixels = acc_map_tensor.size(0) * acc_map_tensor.size(1);
    replace_transparency_by_background_color_kernel<<<num_blocks, num_threads, 0>>>(rgb_map, acc_map, num_pixels);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


__global__ void early_ray_filtering_kernel(
    const float *__restrict__ alphas,
    bool *__restrict__ output_masks,
    const float transmittance_threshold,
    const int *__restrict__ starts,
    const int *__restrict__ ends,
    const int num_rays
)
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (ray_idx < num_rays) {
        
        float my_transmittance = 1.0f;

        int sample_idx = starts[ray_idx];
        int sample_end_idx = ends[ray_idx];
        while (sample_idx < sample_end_idx) {
            float my_alpha = alphas[sample_idx];
            float weight = my_alpha * my_transmittance;
            if (weight < transmittance_threshold) {  // early ray terminate
                output_masks[sample_idx] = false;
            } else {
                output_masks[sample_idx] = true;
            }
            my_transmittance *= 1.0f - my_alpha + 1e-10;
            sample_idx++;
        }

        ray_idx += gridDim.x * blockDim.x;
    }
}


void early_ray_filtering(
    const torch::Tensor& alphas_tensor,
    const torch::Tensor& output_masks_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const float& transmittance_threshold,
    const int& num_rays,
    const int& num_blocks,
    const int& num_threads
)
{
    float *alphas = alphas_tensor.data_ptr<float>();
    bool *output_masks = output_masks_tensor.data_ptr<bool>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();

    early_ray_filtering_kernel<<<num_blocks, num_threads, 0>>>(alphas, output_masks, transmittance_threshold, starts, ends, num_rays);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


__global__ void integrate_filter_kernel(
    const float4 *__restrict__ rgb_sigma,
    float3 *__restrict__ rgb_map,
    float *__restrict__ acc_map,
    float *__restrict__ transmittance,
    const int *__restrict__ starts,
    const int *__restrict__ ends,
    const int *__restrict__ rays_idx,
    const int num_rays
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < num_rays) {
        
        float my_transmittance = 1.0f;
        float3 rgb_out = {0.0f, 0.0f, 0.0f};
        float acc_out = 0.0f;

        int sample_idx = starts[index];
        int sample_end_idx = ends[index];
        while (sample_idx < sample_end_idx) {
            float4 my_rgb_sigma = rgb_sigma[sample_idx];
            float alpha = my_rgb_sigma.w;
            float weight = alpha * my_transmittance;
            my_transmittance *= 1.0f - alpha + 1e-10;
            rgb_out.x += my_rgb_sigma.x * weight;
            rgb_out.y += my_rgb_sigma.y * weight;
            rgb_out.z += my_rgb_sigma.z * weight;
            acc_out += weight;
            sample_idx++;
        }

        int ray_idx = rays_idx[index];

        transmittance[ray_idx] = my_transmittance;
        rgb_map[ray_idx] = rgb_out;
        acc_map[ray_idx] = acc_out;

        index += gridDim.x * blockDim.x;
    }
}


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
)
{
    float4 *rgb_sigma = (float4*)rgb_sigma_tensor.data_ptr<float>();
    float3 *rgb_map = (float3*)rgb_map_tensor.data_ptr<float>();
    float *acc_map = acc_map_tensor.data_ptr<float>();
    float *transmittance = transmittance_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();
    int *rays_idx = rays_idx_tensor.data_ptr<int>();

    integrate_filter_kernel<<<num_blocks, num_threads, 0>>>(rgb_sigma, rgb_map, acc_map, transmittance, starts, ends, rays_idx, num_rays);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}