#include "cuda.h"
#include "stdio.h"
#include "mlp_eval.cuh"
#include "utils.cuh"
#include "cutil_math.cuh"


#define PARAM_SIZE 6212

constexpr int calculate_fourier_embedding_num_output_channels(int num_input_channels, int num_frequencies)
{
    return num_input_channels * (2 * num_frequencies + 1);
}

__device__ __forceinline__ float sigmoid (float x)
{
    return 1.0 / (1.0 + __expf (-x));
}


template<int hidden_dim>
__global__ void mlp_eval_1d_kernel(
    float *__restrict__ output_vectors,
    const float *__restrict__ points,
    const float *__restrict__ viewdirs,
    const float *__restrict__ model_params,
    const int *__restrict__ starts,
    const int *__restrict__ ends
)
{
    constexpr int num_frequencies_position = 10;
    constexpr int num_frequencies_direction = 4;
    
    constexpr int pos_dim = 3;
    constexpr int dir_dim = 3;
    constexpr int pos_embed_dim = calculate_fourier_embedding_num_output_channels(pos_dim, num_frequencies_position);
    constexpr int dir_embed_dim = calculate_fourier_embedding_num_output_channels(dir_dim, num_frequencies_direction);
    constexpr int rgb_dim = 3;
    constexpr int out_dim = 4;

    constexpr float frequency_bands[10] = {1., 2., 4., 8., 16., 32., 64., 128., 256., 512.};

    // for each layer: (#inputs + 1) * #outputs
    constexpr int param_size = (pos_embed_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * (hidden_dim + 1) +
                               (hidden_dim + dir_embed_dim + 1) * hidden_dim + 
                               (hidden_dim + 1) * rgb_dim;

    if (starts[blockIdx.x] == ends[blockIdx.x]) {
        return; // assiociated network does not need to be queried
    }

    // each block handles only one single network
    __shared__ float network_cache[param_size];
    int load_idx = threadIdx.x;
    int network_offset = blockIdx.x * param_size; // block i is reponsible for network i
    while (load_idx < param_size) {
        network_cache[load_idx] = model_params[network_offset + load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();

    int idx = starts[blockIdx.x] + threadIdx.x;
    while (idx < ends[blockIdx.x]) {

        float position[3];
        float direction[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            position[i] = points[idx * pos_dim + i];
            direction[i] = viewdirs[idx * dir_dim + i];
        }

        // Actual network query
        int param_offset = 0;

        // First layer
        float my_hidden_vector_0[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_0[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        /**************************************************************************/
        /* Note: fourier embeddings layout different from KiloNeRF implementation */
        /**************************************************************************/
        // Embedded posistions
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float input_elem = position[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_position; e++) {
                float embedded_input_elem = __sinf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float input_elem = position[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_position; e++) {
                float embedded_input_elem = __cosf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float embedded_input_elem = position[j];
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/

        // Second layer
        float my_hidden_vector_1[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_1[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_1[i] += fmaxf(my_hidden_vector_0[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }

        float density = 0.0;
        density = network_cache[param_offset]; // Bias
        param_offset += 1;
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            density += fmaxf(my_hidden_vector_1[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
            param_offset += 1;
        }

        // Third layer
        float my_hidden_vector_2[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_2[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_2[i] += fmaxf(my_hidden_vector_1[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }

        // Fourth layer: #inputs = hidden_dim + dir_embed_dim
        float my_hidden_vector_3[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_3[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        // Features
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                // offset + 1 because of density, no ReLU (previous layer)
                my_hidden_vector_3[i] += my_hidden_vector_2[j] * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/
        /* Note: fourier embeddings layout different from KiloNeRF implementation */
        /**************************************************************************/
        // Embedded directions
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float input_elem = direction[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_direction; e++) {
                float embedded_input_elem = __sinf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float input_elem = direction[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_direction; e++) {
                float embedded_input_elem = __cosf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float embedded_input_elem = direction[j];
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/

        // Fifth layer
        float my_rgb_vector[rgb_dim];
        #pragma unroll
        for (int i = 0; i < rgb_dim; i++) {
            my_rgb_vector[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < rgb_dim; i++) {
                my_rgb_vector[i] += fmaxf(my_hidden_vector_3[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        
        // Last activation
        #pragma unroll
        for (int i = 0; i < out_dim; i++) {
            float output_elem;
            if (i < rgb_dim) {
                output_elem = sigmoid(my_rgb_vector[i]); // RGB + Sigmoid
            } else {
                output_elem = sigmoid(density);     // Sigmoid for our implementation!!!
            }
            output_vectors[idx * out_dim + i] = output_elem;
        }

        idx += blockDim.x;
    }
}


torch::Tensor mlp_eval_1d(
    const torch::Tensor& points_tensor,
    const torch::Tensor& viewdir_tensor,
    const torch::Tensor& model_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const int& num_networks,
    const int& num_threads
)
{
    float *points = points_tensor.data_ptr<float>();
    float *viewdirs = viewdir_tensor.data_ptr<float>();
    float *model = model_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();

    int batch_size = points_tensor.size(0);
    torch::Tensor output_vectors_tensor = torch::zeros({batch_size, 4}, points_tensor.options());
    float *output_vectors = output_vectors_tensor.data_ptr<float>();
    
    mlp_eval_1d_kernel<32><<<num_networks, num_threads, 0>>>(output_vectors, points, viewdirs, model, starts, ends);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return output_vectors_tensor;
}


template<int hidden_dim>
__global__ void mlp_eval_1d_filter_kernel(
    float *__restrict__ output_vectors,
    const float *__restrict__ points,
    const float *__restrict__ viewdirs,
    const float *__restrict__ model_params,
    const int *__restrict__ starts,
    const int *__restrict__ ends,
    const int *__restrict__ planes_idx
)
{
    constexpr int num_frequencies_position = 10;
    constexpr int num_frequencies_direction = 4;
    
    constexpr int pos_dim = 3;
    constexpr int dir_dim = 3;
    constexpr int pos_embed_dim = calculate_fourier_embedding_num_output_channels(pos_dim, num_frequencies_position);
    constexpr int dir_embed_dim = calculate_fourier_embedding_num_output_channels(dir_dim, num_frequencies_direction);
    constexpr int rgb_dim = 3;
    constexpr int out_dim = 4;

    constexpr float frequency_bands[10] = {1., 2., 4., 8., 16., 32., 64., 128., 256., 512.};

    // for each layer: (#inputs + 1) * #outputs
    constexpr int param_size = (pos_embed_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * hidden_dim +
                               (hidden_dim + 1) * (hidden_dim + 1) +
                               (hidden_dim + dir_embed_dim + 1) * hidden_dim + 
                               (hidden_dim + 1) * rgb_dim;

    if (starts[blockIdx.x] == ends[blockIdx.x]) {
        return; // assiociated network does not need to be queried
    }

    int plane_idx = planes_idx[blockIdx.x];

    // each block handles only one single network
    __shared__ float network_cache[param_size];
    int load_idx = threadIdx.x;
    int network_offset = plane_idx * param_size; // block i is reponsible for network i
    while (load_idx < param_size) {
        network_cache[load_idx] = model_params[network_offset + load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();

    int idx = starts[blockIdx.x] + threadIdx.x;
    while (idx < ends[blockIdx.x]) {

        float position[3];
        float direction[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            position[i] = points[idx * pos_dim + i];
            direction[i] = viewdirs[idx * dir_dim + i];
        }

        // Actual network query
        int param_offset = 0;

        // First layer
        float my_hidden_vector_0[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_0[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        /**************************************************************************/
        /* Note: fourier embeddings layout different from KiloNeRF implementation */
        /**************************************************************************/
        // Embedded posistions
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float input_elem = position[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_position; e++) {
                float embedded_input_elem = __sinf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float input_elem = position[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_position; e++) {
                float embedded_input_elem = __cosf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < pos_dim; j++) {
            float embedded_input_elem = position[j];
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_0[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/

        // Second layer
        float my_hidden_vector_1[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_1[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_1[i] += fmaxf(my_hidden_vector_0[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }

        float density = 0.0;
        density = network_cache[param_offset]; // Bias
        param_offset += 1;
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            density += fmaxf(my_hidden_vector_1[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
            param_offset += 1;
        }

        // Third layer
        float my_hidden_vector_2[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_2[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_2[i] += fmaxf(my_hidden_vector_1[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }

        // Fourth layer: #inputs = hidden_dim + dir_embed_dim
        float my_hidden_vector_3[hidden_dim];
        #pragma unroll
        for (int i = 0; i < hidden_dim; i++) {
            my_hidden_vector_3[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        // Features
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                // offset + 1 because of density, no ReLU (previous layer)
                my_hidden_vector_3[i] += my_hidden_vector_2[j] * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/
        /* Note: fourier embeddings layout different from KiloNeRF implementation */
        /**************************************************************************/
        // Embedded directions
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float input_elem = direction[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_direction; e++) {
                float embedded_input_elem = __sinf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float input_elem = direction[j];
            #pragma unroll
            for (int e = 0; e < num_frequencies_direction; e++) {
                float embedded_input_elem = __cosf(frequency_bands[e] * input_elem);
                #pragma unroll
                for (int i = 0; i < hidden_dim; i++) {
                    my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                    param_offset += 1;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < dir_dim; j++) {
            float embedded_input_elem = direction[j];
            #pragma unroll
            for (int i = 0; i < hidden_dim; i++) {
                my_hidden_vector_3[i] += embedded_input_elem * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        /**************************************************************************/

        // Fifth layer
        float my_rgb_vector[rgb_dim];
        #pragma unroll
        for (int i = 0; i < rgb_dim; i++) {
            my_rgb_vector[i] = network_cache[param_offset]; // Bias
            param_offset += 1;
        }
        #pragma unroll
        for (int j = 0; j < hidden_dim; j++) {
            #pragma unroll
            for (int i = 0; i < rgb_dim; i++) {
                my_rgb_vector[i] += fmaxf(my_hidden_vector_3[j], 0.0f) * network_cache[param_offset]; // Vector Matrix Multiplication
                param_offset += 1;
            }
        }
        
        // Last activation
        #pragma unroll
        for (int i = 0; i < out_dim; i++) {
            float output_elem;
            if (i < rgb_dim) {
                output_elem = sigmoid(my_rgb_vector[i]); // RGB + Sigmoid
            } else {
                output_elem = sigmoid(density);     // Sigmoid for our implementation!!!
            }
            output_vectors[idx * out_dim + i] = output_elem;
        }

        idx += blockDim.x;
    }
}


torch::Tensor mlp_eval_1d_filter(
    const torch::Tensor& points_tensor,
    const torch::Tensor& viewdir_tensor,
    const torch::Tensor& model_tensor,
    const torch::Tensor& starts_tensor,
    const torch::Tensor& ends_tensor,
    const torch::Tensor& planes_idx_tensor,
    const int& num_networks,
    const int& num_threads
)
{
    float *points = points_tensor.data_ptr<float>();
    float *viewdirs = viewdir_tensor.data_ptr<float>();
    float *model = model_tensor.data_ptr<float>();
    int *starts = starts_tensor.data_ptr<int>();
    int *ends = ends_tensor.data_ptr<int>();
    int *planes_idx = planes_idx_tensor.data_ptr<int>();

    int batch_size = points_tensor.size(0);
    torch::Tensor output_vectors_tensor = torch::zeros({batch_size, 4}, points_tensor.options());
    float *output_vectors = output_vectors_tensor.data_ptr<float>();
    
    mlp_eval_1d_filter_kernel<32><<<num_networks, num_threads, 0>>>(output_vectors, points, viewdirs, model, starts, ends, planes_idx);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    return output_vectors_tensor;
}