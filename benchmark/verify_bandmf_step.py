"""
Sanity check: CorNoiseGenerator.step() computes the correct recurrence on CPU,
and the output transfers to GPU correctly.

The recurrence (Algorithm 9, BandMF):
    x_t = z_t - sum_{j in [t-bw+1, t-1]} (C[t,j] / C[t,t]) * x_j

Usage:
    python verify_bandmf_step.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastDP.bandmf import CorNoiseGenerator


def run():
    bw = 3
    C = np.array([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 3.0, 0.0, 0.0, 0.0],
        [0.3, 0.7, 2.5, 0.0, 0.0],
        [0.0, 0.4, 0.6, 3.5, 0.0],
        [0.0, 0.0, 0.5, 0.8, 2.0],
    ], dtype=np.float32)
    z_vals = [1.0, 2.0, 3.0, 4.0, 5.0] # Gaussian noise pre-divided by C[i, i]

    # Reference: explicit recurrence computed by hand
    x_ref = []
    for t, z_t in enumerate(z_vals):
        acc = z_t
        for j in range(max(0, t - bw + 1), t):
            acc -= (C[t, j] / C[t, t]) * x_ref[j]
        x_ref.append(acc)

    # CorNoiseGenerator on CPU
    gen = CorNoiseGenerator(C, bandwidth=bw, device='cpu', partition=1)
    assert gen._matrix.device.type == 'cpu', f"matrix on wrong device: {gen._matrix.device}"

    x_gen = []
    for z_t in z_vals:
        inp = torch.tensor([z_t])
        assert inp.device.type == 'cpu'
        out = gen.step(inp)
        x_gen.append(out.item())
        gen.advance()

    # Compare
    print(f"  {'step':>4}  {'z':>6}  {'x_ref':>10}  {'x_gen (CPU)':>12}  {'err':>10}")
    print(f"  {'-'*50}")
    ok = True
    for t, (z, xr, xg) in enumerate(zip(z_vals, x_ref, x_gen)):
        err = abs(xr - xg)
        ok = ok and err < 1e-5
        print(f"  {t:>4}  {z:>6.1f}  {xr:>10.6f}  {xg:>12.6f}  {err:>10.2e}  {'OK' if err < 1e-5 else 'FAIL'}")

    # CPU → GPU transfer check
    sample = torch.tensor(x_gen)
    assert sample.device.type == 'cpu'
    sample_gpu = sample.to('cuda:0', non_blocking=True)
    assert sample_gpu.device.type == 'cuda'
    roundtrip_err = (sample_gpu.cpu() - sample).abs().max().item()
    ok = ok and roundtrip_err < 1e-6
    print(f"\n  CPU→GPU roundtrip err: {roundtrip_err:.2e}  "
          f"({'OK' if roundtrip_err < 1e-6 else 'FAIL'})")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
