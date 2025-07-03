from torch.utils.cpp_extension import load
import torch.nn.functional as F
import time
import torch
import timeit

relu_ext = load(name="relu_cpu_simple", sources=["activations_cpu/relu.cpp"],
                    extra_cflags=['-mavx', '-O3'],)  # Enable AVX, optimize)


dummy = torch.randn((1000, 1000))

# tic = time.time()
# for _ in range(100):
#     relu_ext.relu_cpu(dummy)
# toc = time.time()
# print(f"Duration: {toc - tic}")


# tic = time.time()
# for _ in range(100):
#     F.relu(dummy)
# toc = time.time()
# print(f"Duration: {toc - tic}")


print(timeit.timeit(lambda: relu_ext.relu_cpu(dummy), number=100))
print(timeit.timeit(lambda: F.relu(dummy), number=100))
