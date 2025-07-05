#include <torch/extension.h>

torch::Tensor relu(torch::Tensor input, bool in_place = false);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("relu", &relu, "ReLU activation", py::arg("input"), py::arg("in_place") = false);
}