#ifndef _KNN_DISTANCE
#define _KNN_DISTANCE

#include <vector>
#include <torch/extension.h>

//CUDA declarations
std::vector<at::Tensor> KNNDistance(
    const at::Tensor query_xyz,
    const at::Tensor key_xyz,
    const int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_distance", &KNNDistance, "k-nearest neighbor with distance (CUDA)");
}

#endif
