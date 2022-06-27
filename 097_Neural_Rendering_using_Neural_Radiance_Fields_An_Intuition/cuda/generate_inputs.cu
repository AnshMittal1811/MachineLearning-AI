#include "generate_inputs.cuh"
#include "utils.cuh"
#include "cuda.h"

__device__ __constant__ float c_c2w[9];

__global__ void get_rays_d_kernel(
    const int H,
    const int W,
    const float px,
    const float py,
    const float fx,
    const float fy,
    const float *__restrict__ ndc_points,
    float *__restrict__ rays_d
)
{
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx_x < W) {
        int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
        while (idx_y < H) {

            float in[3];
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                in[i] = ndc_points[(idx_x + idx_y * W) * 3 + i];
            }

            in[0] = (in[0] - px) / fx;
            in[1] = (in[1] - py) / fy;

            float out[3] = {0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    out[i] += in[j] * c_c2w[i * 3 + j];
                }
            }

            // Normalize direction
            float norm = 0.0f;
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                norm += out[i] * out[i];
            }
            norm = sqrtf(norm);
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                out[i] /= norm;
            }
        
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                rays_d[(idx_x + idx_y * W) * 3 + i] = out[i];
            }
            idx_y += gridDim.y * blockDim.y; 
        }
        idx_x += gridDim.x * blockDim.x;
    }
}


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
)
{
    torch::Tensor rays_d_tensor = torch::empty({H, W, 3}, c2w_tensor.options());
    float *rays_d = rays_d_tensor.data_ptr<float>();
    float *c2w = c2w_tensor.data_ptr<float>();
    cudaMemcpyToSymbol(c_c2w, c2w, 9 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    float *ndc_points = ndc_points_tensor.data_ptr<float>();

    dim3 gridDim(root_num_blocks, root_num_blocks);
    dim3 blockDim(root_num_threads, root_num_threads);
    get_rays_d_kernel<<<gridDim, blockDim, 0>>>(H, W, px, py, fx, fy, ndc_points, rays_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return rays_d_tensor;
}


__device__ void cross_product(float vect_A[], float vect_B[], float cross_P[])
{
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
}


__global__ void compute_ray_plane_intersection_mt_kernel(
    const float *__restrict__ rays,
    const float *__restrict__ camera_center,
    const float *__restrict__ planes_vertices,
    bool *__restrict__ hits,
    const int num_rays
)
{
    // Loading input
    int plane_idx = blockIdx.x;
    float O[3], A[3], B[3], C[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        O[i] = camera_center[i];
        A[i] = planes_vertices[plane_idx * 3 * 3 + 0 + i];
        B[i] = planes_vertices[plane_idx * 3 * 3 + 3 + i];
        C[i] = planes_vertices[plane_idx * 3 * 3 + 6 + i];
    }

    int ray_idx = threadIdx.x;
    while (ray_idx < num_rays) {

        float ray_d[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            ray_d[i] = rays[ray_idx * 3 + i];
        }

        float T[3], E1[3], E2[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            T[i] = O[i] - A[i];
            E1[i] = B[i] - A[i];
            E2[i] = C[i] - A[i];
        }

        float P[3], Q[3];
        cross_product(ray_d, E2, P);
        cross_product(T, E1, Q);

        float den = 0.0;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            den += P[i] * E1[i];
        }

        // filter tiny denominator value
        // float eps = 1e-5;
        // den = den >= 0 && den < eps ? eps : den;
        // den = den < 0 && den > -eps ? -eps : den;

        float den_inv = 1 / den;

        float t = 0.0;
        float u = 0.0;
        float v = 0.0;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            t += Q[i] * E2[i];
            u += P[i] * T[i];
            v += Q[i] * ray_d[i];
        }
        t *= den_inv;
        u *= den_inv;
        v *= den_inv;

        int hit_idx = plane_idx * num_rays + ray_idx;
        if (u >= 0 && u <= 1 && v >= 0 && v <= 1 && t >= 0) {  // t < 0: negative depth
            hits[hit_idx] = true;
        } else {
            hits[hit_idx] = false;
        }

        ray_idx += blockDim.x;
    }
}

/*
Inputs:
- plane_vertices (plane_num x 3 x 3)
- rays           (pixels_num x 3)
- camera_center  (3)
Outputs:
- intersection (plane_num x pixels_num)
*/
void compute_ray_plane_intersection_mt(
    const torch::Tensor& planes_vertices_tensor,
    const torch::Tensor& rays_tensor,
    const torch::Tensor& camera_center_tensor,
    const torch::Tensor& hits_tensor,
    const int& num_blocks,
    const int& num_threads
)
{
    float *rays = rays_tensor.data_ptr<float>();
    float *planes_vertices = planes_vertices_tensor.data_ptr<float>();
    float *camera_center = camera_center_tensor.data_ptr<float>();

    int n_rays = rays_tensor.size(0);
    int n_planes = planes_vertices_tensor.size(0);
    bool *hits = hits_tensor.data_ptr<bool>();

    compute_ray_plane_intersection_mt_kernel<<<num_blocks, num_threads, 0>>>(rays, camera_center, planes_vertices, hits, n_rays);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


__global__ void store_ray_plane_intersection_mt_kernel(
    const float *__restrict__ rays,
    const int *__restrict__ rays_idx,
    const float *__restrict__ camera_center,
    const float *__restrict__ planes_vertices,
    const int *__restrict__ starts,
    const int *__restrict__ ends,
    float *__restrict__ points,
    float *__restrict__ view_dirs,
    float *__restrict__ depths
)
{
    // Loading input
    int plane_idx = blockIdx.x;
    float O[3], A[3], B[3], C[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        O[i] = camera_center[i];
        A[i] = planes_vertices[plane_idx * 3 * 3 + 0 + i];
        B[i] = planes_vertices[plane_idx * 3 * 3 + 3 + i];
        C[i] = planes_vertices[plane_idx * 3 * 3 + 6 + i];
    }

    int idx = starts[blockIdx.x] + threadIdx.x;
    while (idx < ends[blockIdx.x]) {

        int ray_idx = rays_idx[idx];

        float ray_d[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            ray_d[i] = rays[ray_idx * 3 + i];
        }

        float T[3], E1[3], E2[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            T[i] = O[i] - A[i];
            E1[i] = B[i] - A[i];
            E2[i] = C[i] - A[i];
        }

        float P[3], Q[3];
        cross_product(ray_d, E2, P);
        cross_product(T, E1, Q);

        float den = 0.0;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            den += P[i] * E1[i];
        }

        // filter tiny denominator value
        // float eps = 1e-5;
        // den = den >= 0 && den < eps ? eps : den;
        // den = den < 0 && den > -eps ? -eps : den;

        float den_inv = 1 / den;

        float t = 0.0;
        float u = 0.0;
        float v = 0.0;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            t += Q[i] * E2[i];
            u += P[i] * T[i];
            v += Q[i] * ray_d[i];
        }
        t *= den_inv;
        u *= den_inv;
        v *= den_inv;

        // saving the result
        for (int i = 0; i < 3; i++) {
            int save_idx = idx * 3 + i;
            points[save_idx] = O[i] + t * ray_d[i];
            view_dirs[save_idx] = ray_d[i];
        }
        depths[idx] = t;

        idx += blockDim.x;
    }
}


/*
Inputs:
- plane_vertices (plane_num x 3 x 3)
- rays           (pixels_num x 3)
- camera_center  (3)
- intersection   (plane_num x pixels_num) <- maybe not ?
- rays_idx       (n_points)
Outputs:
- points_tensor  (n_points, 3)
- viewdir_tensor (n_points, 3)
- depth_tensor   (n_points)
*/
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
)
{
    float *planes_vertices = planes_vertices_tensor.data_ptr<float>();
    float *rays = rays_tensor.data_ptr<float>();
    int *rays_idx = rays_idx_tensor.data_ptr<int>();
    float *camera_center = camera_center_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();
    float *points = points_tensor.data_ptr<float>();
    float *view_dirs = view_dirs_tensor.data_ptr<float>();
    float *depths = depths_tensor.data_ptr<float>();

    store_ray_plane_intersection_mt_kernel<<<num_blocks, num_threads, 0>>>(rays, rays_idx, camera_center, planes_vertices, starts, ends, points, view_dirs, depths);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


__global__ void compute_ray_plane_intersection_kernel(
    const float *__restrict__ rays,
    const float *__restrict__ camera_center,
    const float *__restrict__ planes_center,
    const float *__restrict__ planes_normal,
    const float *__restrict__ planes_x,
    const float *__restrict__ planes_y,
    const float *__restrict__ planes_h,
    const float *__restrict__ planes_w,
    bool *__restrict__ hits,
    const int num_rays
)
{
    // Loading plane parameter
    int plane_idx = blockIdx.x;
    float plane_h = planes_h[plane_idx];
    float plane_w = planes_w[plane_idx];
    float plane_center[3], plane_normal[3], plane_x[3], plane_y[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        plane_center[i] = planes_center[plane_idx * 3 + i];
        plane_normal[i] = planes_normal[plane_idx * 3 + i];
        plane_x[i] = planes_x[plane_idx * 3 + i];
        plane_y[i] = planes_y[plane_idx * 3 + i];
    }

    // Loading camera origin
    float cam_center[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        cam_center[i] = camera_center[i];
    }

    int ray_idx = threadIdx.x;
    while (ray_idx < num_rays) {

        float ray_d[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            ray_d[i] = rays[ray_idx * 3 + i];
        }

        // (P_k - O) dot N_k
        float numerator = 0.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            numerator += (plane_center[i] - cam_center[i]) * plane_normal[i];
        }

        // d dot N_k
        float denominator = 0.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            denominator += ray_d[i] * plane_normal[i];
        }

        // filter tiny denominator value
        float eps = 1e-5;
        denominator = denominator >= 0 && denominator < eps ? eps : denominator;
        denominator = denominator < 0 && denominator > -eps ? -eps : denominator;

        float t = numerator / denominator;  // this is 'depth'
        // x = o + td
        float point_hit[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            point_hit[i] = cam_center[i] + t * ray_d[i];
        }

        /*** project to plane space & check if it hits with the plane ***/
        float point_on_plane[2] = {0.0, 0.0};
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            point_on_plane[0] += (point_hit[i] - plane_center[i]) * plane_x[i];
            point_on_plane[1] += (point_hit[i] - plane_center[i]) * plane_y[i];
        }

        // normalize
        float point_x = point_on_plane[0] * 2 / plane_w;
        float point_y = point_on_plane[1] * 2 / plane_h;

        int hit_idx = plane_idx * num_rays + ray_idx;
        if (point_x >= -1 && point_x <= 1 && point_y >= -1 && point_y <= 1 && t >= 0) {  // t < 0: negative depth
            hits[hit_idx] = true;
        } else {
            hits[hit_idx] = false;
        }

        ray_idx += blockDim.x;
    }
}



/*
Inputs:
- plane_center  (plane_num x 3)
- plane_normal  (plane_num x 3)
- plane_x_basis (plane_num x 3)
- plane_y_basis (plane_num x 3)
- plane_height  (plane_num)
- plane_width   (plane_num)
- rays          (pixels_num x 3)
- camera_center (3)
Outputs:
- intersection (plane_num x pixels_num)
*/
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
)
{
    float *rays = rays_tensor.data_ptr<float>();
    float *planes_center = planes_center_tensor.data_ptr<float>();
    float *planes_normal = planes_normal_tensor.data_ptr<float>();
    float *planes_x = planes_x_tensor.data_ptr<float>();
    float *planes_y = planes_y_tensor.data_ptr<float>();
    float *planes_h = planes_h_tensor.data_ptr<float>();
    float *planes_w = planes_w_tensor.data_ptr<float>();
    float *camera_center = camera_center_tensor.data_ptr<float>();

    int n_rays = rays_tensor.size(0);
    int n_planes = planes_h_tensor.size(0);
    bool *hits = hits_tensor.data_ptr<bool>();

    compute_ray_plane_intersection_kernel<<<num_blocks, num_threads, 0>>>(rays, camera_center, planes_center, planes_normal, planes_x, planes_y, planes_h, planes_w, hits, n_rays);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}