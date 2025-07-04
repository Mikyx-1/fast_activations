#include <torch/extension.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <omp.h>        // OpenMP

torch::Tensor relu_cpu(torch::Tensor input, const bool in_place = false) {
    TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    float* __restrict__ in_data = input.data_ptr<float>();
    const int N = input.numel();

    if (in_place) {
        #pragma omp parallel
        {
            __m256 zero = _mm256_setzero_ps();

            #pragma omp for
            for (int i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    __m256 x = _mm256_loadu_ps(in_data + i);
                    __m256 y = _mm256_max_ps(x, zero);
                    _mm256_storeu_ps(in_data + i, y);
                } else {
                    // tail handling
                    for (int j = i; j < N; ++j) {
                        // in_data[j] = in_data[j] > 0.0f ? in_data[j] : 0.0f;
                        in_data[j] = std::max(0.f, in_data[j]);
                    }
                }
            }
        }

        return input;

    } else {
        torch::Tensor output = torch::empty_like(input);
        float* __restrict__ out_data = output.data_ptr<float>();

        #pragma omp parallel
        {
            __m256 zero = _mm256_setzero_ps();

            #pragma omp for
            for (int i = 0; i < N; i += 8) {
                if (i + 8 <= N) {
                    __m256 x = _mm256_loadu_ps(in_data + i);
                    __m256 y = _mm256_max_ps(x, zero);
                    _mm256_storeu_ps(out_data + i, y);
                } else {
                    // tail handling
                    for (int j = i; j < N; ++j) {
                        // out_data[j] = in_data[j] > 0.0f ? in_data[j] : 0.0f;
                        out_data[j] = std::max(0.f, in_data[j]);
                    }
                }
            }
        }

        return output;
    }
}

// Pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_cpu", &relu_cpu, "SIMD + OpenMP CPU ReLU",
          pybind11::arg("input"),
          pybind11::arg("in_place") = false);
}
