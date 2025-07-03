from torch.utils.cpp_extension import load
import torch.nn.functional as F
import torch
import timeit

relu_ext = load(
    name="relu_cpu_simple",
    sources=["activations_cpu/relu.cpp"],
    extra_cflags=["-mavx2", "-O3", "-fopenmp", "-march=native"],
    extra_ldflags=["-fopenmp"],
)

dummy = torch.randn((10000, 10000))

print("Our custom ReLU:")
print(timeit.timeit(lambda: relu_ext.relu_cpu(dummy), number=10))

print("PyTorch F.relu:")
print(timeit.timeit(lambda: F.relu(dummy), number=10))
