#ifndef _CHAMFER
#define _CHAMFER

#include <vector>
#include <torch/extension.h>

//CUDA declarations
std::vector<at::Tensor> ChamferForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2);

std::vector<at::Tensor> ChamferBackward(
    const at::Tensor grad_dist1,
    const at::Tensor grad_dist2,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor idx1,
    const at::Tensor idx2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chamfer_forward", &ChamferForward,"Chamfer distance forward (CUDA)");
  m.def("chamfer_backward", &ChamferBackward, "Chamfer distance backward (CUDA)");
}

#endif
