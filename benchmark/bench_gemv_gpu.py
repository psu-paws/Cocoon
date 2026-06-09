"""
Benchmark: GPU-native BandMF GEMV throughput.

All computation and noise stay on GPU — no CPU subprocess, no H2D transfer.
Measures the time for forward substitution (BandMF GEMV) on a CUDA tensor.

Usage:
    python bench_gemv_gpu.py <num_params> <bandwidth> [partition=1]
Example:
    python bench_gemv_gpu.py 6000000 32
    python bench_gemv_gpu.py 6000000 32 4
"""
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastDP.bandmf import CorNoiseGenerator


def run_benchmark(tensor_size, iterations, band_size, partition):
    rng = np.random.default_rng(42)
    C_np = np.tril(rng.standard_normal((1000, 1000)).astype(np.float32))
    np.fill_diagonal(C_np, np.abs(np.diag(C_np)) + 0.5)

    unit_size = (tensor_size + partition - 1) // partition
    forward = CorNoiseGenerator(C_np, bandwidth=band_size, device='cuda', partition=partition)
    forward.skip_to_steady_state()

    buf_bytes = (band_size - 1) * unit_size * 4
    print(f"  unit_size={unit_size:,}  buf_size={buf_bytes/1e6:.1f} MB per partition")

    gemv_starts = []
    gemv_ends   = []

    for i in range(iterations):
        noise = torch.normal(mean=0.0, std=1.0, size=(unit_size,), device='cuda')

        gemv_starts.append(torch.cuda.Event(enable_timing=True))
        gemv_starts[-1].record()
        for j in range(partition):
            _ = forward.step(noise, j)
        forward.advance()
        gemv_ends.append(torch.cuda.Event(enable_timing=True))
        gemv_ends[-1].record()

        if (i + 1) % 100 == 0 or iterations <= 10:
            print(f"  iter {i+1}/{iterations}", flush=True)

    torch.cuda.synchronize()
    gemv_ms = [s.elapsed_time(e) for s, e in zip(gemv_starts, gemv_ends)]

    avg_ms = np.mean(gemv_ms)
    std_ms = np.std(gemv_ms)
    bw_gbs = buf_bytes * partition / (avg_ms * 1e-3) / 1e9

    print(f"\nGPU GEMV ({partition} partition{'s' if partition > 1 else ''}/step):")
    print(f"  avg {avg_ms:.3f} ms  std {std_ms:.3f} ms")
    print(f"  effective BW: {bw_gbs:.2f} GB/s ")


def main():
    tensor_size = int(sys.argv[1])
    band_size   = int(sys.argv[2])
    partition   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    iterations  = 300

    print(f"tensor_size={tensor_size:,}  bandwidth={band_size}  partition={partition}")
    run_benchmark(tensor_size, iterations, band_size, partition)


if __name__ == "__main__":
    main()
