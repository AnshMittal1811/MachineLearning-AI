#ifndef _NN_DISTANCE
#define _NN_DISTANCE

#include <vector>
#include <torch/extension.h>

//CUDA declarations
std::vector<at::Tensor> NNDistanceForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2);

std::vector<at::Tensor> NNDistanceBackward(
    const at::Tensor grad_dist1,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor idx1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nn_distance_forward", &NNDistanceForward,"NN distance forward (CUDA)");
  m.def("nn_distance_backward", &NNDistanceBackward, "NN distance backward (CUDA)");
}

#endif
