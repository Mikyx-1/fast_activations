import fast_activations
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dummy = torch.randn((2000, 10000), device=device)

NUM_ITERS = 10000
in_place = True

tic = time.time()
for _ in range(NUM_ITERS):
    fast_activations.relu(dummy, in_place)
toc = time.time()

torch.cuda.synchronize()

print("\n=== CUDA ===\n")
print(f"Duration: {toc - tic}")


tic = time.time()
for _ in range(NUM_ITERS):
    res1 = F.relu(dummy, in_place)
toc = time.time()

torch.cuda.synchronize()

print("\n=== Pytorch ===\n")
print(f"Duration: {toc - tic}")


# print(f"all_close: {torch.allclose(res, res1)}, Max diff: {torch.abs(res - res1).max()}")