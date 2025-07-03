// main.cpp
#include <torch/extension.h>


torch::Tensor relu_cpu(torch::Tensor input, bool in_place = false)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", torch::wrap_pybind_function(relu_unified), "Unified ReLU (CPU/GPU)");
}