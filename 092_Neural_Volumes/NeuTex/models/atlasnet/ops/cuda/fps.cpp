#ifndef _FPS
#define _FPS

#include <torch/extension.h>

// CUDA declarations
at::Tensor FarthestPointSample(
    const at::Tensor points,
    const int64_t num_centroids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("farthest_point_sample", &FarthestPointSample, "Farthest point sampling (CUDA)");
}

#endif