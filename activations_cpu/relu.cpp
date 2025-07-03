#include <torch/extension.h>



torch::Tensor relu_cpu(torch::Tensor input, bool in_place = false)
{
    TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    if (in_place){
        float* data = input.data_ptr<float>();
        int N = input.numel();
        for (int i = 0; i < N; ++i)
        {
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
        return input;
    } else{
        torch::Tensor output = torch::empty_like(input);
        float* in_data = input.data_ptr<float>();
        float* out_data = output.data_ptr<float>();
        int N = input.numel();
        for (int i = 0; i < N; ++i)
        {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;
        }

        return output;

    }

}


// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_cpu", &relu_cpu, "Simple CPU ReLU (no SIMD, no unrolling)",
          pybind11::arg("input"),
          pybind11::arg("in_place") = false);
}