#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> BallQueryForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const float radius);

std::vector<at::Tensor> BallQueryBackward(
    const at::Tensor grad_dist,
    const at::Tensor count,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const float radius);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_forward", &BallQueryForward, "Ball query forward (CUDA)");
  m.def("ball_query_backward", &BallQueryBackward, "Ball query backward (CUDA)");
}

#endif