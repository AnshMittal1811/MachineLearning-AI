#ifndef _GROUP_POINTS
#define _GROUP_POINTS

#include <torch/extension.h>

// CUDA declarations
at::Tensor GroupPointsForward(
    const at::Tensor input,
    const at::Tensor index);

at::Tensor GroupPointsBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const int64_t num_points);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("group_points_forward", &GroupPointsForward, "Group points forward (CUDA)");
  m.def("group_points_backward", &GroupPointsBackward, "Group points backward (CUDA)");
}

#endif