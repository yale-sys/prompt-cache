# On RTX 4090
# Host-to-Host (CPU to CPU) Average Latency: 3.79 milliseconds
# Host-to-Device (CPU to GPU) Average Latency: 5.34 milliseconds
# Device-to-Device (GPU to GPU) Average Latency: 0.23 milliseconds
# Device-to-Host (GPU to CPU) Average Latency: 5.88 milliseconds


import torch
import time

NUM_LAYERS = 30
SEQ_LEN = 5000
CACHE_DIM = (40, SEQ_LEN, 128)

print('loaded')


def create_cache(device):
    return [(torch.rand(CACHE_DIM, dtype=torch.float16, device=device),
             torch.rand(CACHE_DIM, dtype=torch.float16, device=device)) for _ in
            range(NUM_LAYERS)]


def benchmark_transfer(src_cache, dst_cache, description):
    start_time = time.time()
    for src, dst in zip(src_cache, dst_cache):
        dst[0].copy_(src[0], non_blocking=True)
        dst[1].copy_(src[0], non_blocking=True)
    torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
    elapsed = (time.time() - start_time) / NUM_LAYERS
    print(f"{description} Average Latency: {elapsed * 1000:.2f} milliseconds")


cpu_cache = create_cache('cpu')
gpu_cache = create_cache('cuda')
cpu_cache_clone = create_cache('cpu')
gpu_cache_clone = create_cache('cuda')

# Host-to-Host (CPU to CPU) Transfer
benchmark_transfer(cpu_cache, cpu_cache_clone, "Host-to-Host (CPU to CPU)")

# Host-to-Device (CPU to GPU) Transfer
benchmark_transfer(cpu_cache, gpu_cache_clone, "Host-to-Device (CPU to GPU)")

# Device-to-Device (GPU to GPU) Transfer
benchmark_transfer(gpu_cache, gpu_cache_clone, "Device-to-Device (GPU to GPU)")

# Device-to-Host (GPU to CPU) Transfer
benchmark_transfer(gpu_cache, cpu_cache_clone, "Device-to-Host (GPU to CPU)")
