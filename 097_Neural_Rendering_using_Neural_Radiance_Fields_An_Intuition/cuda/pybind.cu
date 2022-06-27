#include "utils.cuh"
#include "mlp_eval.cuh"
#include "generate_inputs.cuh"
#include "reorder.cuh"
#include "integrate.cuh"
#include "prebake.cuh"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mlp_eval_1d", &mlp_eval_1d, "");
    m.def("get_rays_d", &get_rays_d, "");
    m.def("integrate", &integrate, "");
    m.def("replace_transparency_by_background_color", &replace_transparency_by_background_color, "");
    m.def("compute_ray_plane_intersection", &compute_ray_plane_intersection, "");
    m.def("compute_ray_plane_intersection_mt", &compute_ray_plane_intersection_mt, "");
    m.def("store_ray_plane_intersection_mt", &store_ray_plane_intersection_mt, "");   

    m.def("sort_by_key_int16_int64", &sort_by_key_int16_int64, "");
    m.def("sort_by_key_int32_int64", &sort_by_key_int32_int64, "");
    m.def("sort_by_key_int64_int64", &sort_by_key_int64_int64, "");
    m.def("sort_by_key_float32_int64", &sort_by_key_float32_int64, "");

    m.def("sample_from_planes_alpha", &sample_from_planes_alpha, "");
    m.def("early_ray_filtering", &early_ray_filtering, "");

    m.def("mlp_eval_1d_filter", &mlp_eval_1d_filter, "");
    m.def("integrate_filter", &integrate_filter, "");
}