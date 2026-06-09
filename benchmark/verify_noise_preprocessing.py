"""
Verify that NoiseCache preprocessing == naive per-step BandMF accumulation.
Expected: max |naive - preprocessing| < 1e-5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from fastDP.bandmf import CorNoiseGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_banded_lower_triangular(n: int, bandwidth: int, seed: int = 42) -> np.ndarray:
    """Random banded lower-triangular matrix with positive diagonal."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(max(0, i - bandwidth + 1), i):
            C[i, j] = float(rng.standard_normal() * 0.5)
        C[i, i] = float(abs(rng.standard_normal()) + 1.0)
    return C


def make_access_pattern(n_rows: int, n_steps: int, seed: int = 1) -> list[list[int]]:
    """Random access pattern: each step accesses 2–5 distinct rows."""
    rng = np.random.default_rng(seed)
    return [
        sorted(rng.choice(n_rows, size=rng.integers(2, 6), replace=False).tolist())
        for _ in range(n_steps)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Hyperparameters
    n_rows    = 20
    m_spa     = 8
    n_steps   = 16
    bandwidth = 8
    threshold = 3
    noise_std = 1.0

    print("=" * 60)
    print("NoiseCache preprocessing equivalence verification")
    print("=" * 60)
    print(f"  n_rows={n_rows}, m_spa={m_spa}, n_steps={n_steps}")
    print(f"  bandwidth={bandwidth}, threshold={threshold}")
    print()

    C = make_banded_lower_triangular(n_steps, bandwidth, seed=42)
    diag_vals = np.array([C[j, j] for j in range(n_steps)], dtype=np.float32)

    access_pattern = make_access_pattern(n_rows, n_steps, seed=1)

    access_counts = np.zeros(n_rows, dtype=int)
    for step_rows in access_pattern:
        for r in step_rows:
            access_counts[r] += 1

    cold_indices = np.where(access_counts <= threshold)[0]
    hot_indices  = np.where(access_counts >  threshold)[0]
    n_cold = len(cold_indices)
    cold_set = set(cold_indices.tolist())
    cold_local = {r: idx for idx, r in enumerate(cold_indices.tolist())}

    print(f"  Per-row access counts: {access_counts.tolist()}")
    print(f"  Cold rows (≤{threshold} accesses): {cold_indices.tolist()}")
    print(f"  Hot  rows (>{threshold} accesses): {hot_indices.tolist()}")
    print(f"  n_cold={n_cold}, n_hot={len(hot_indices)}")
    print()

    torch.manual_seed(123)
    z_all = []
    for j in range(n_steps):
        z_j = torch.normal(mean=0.0,
                           std=float(noise_std / diag_vals[j]),
                           size=(n_rows * m_spa,))
        z_all.append(z_j)

    # -----------------------------------------------------------------------
    # Naive simulation
    # -----------------------------------------------------------------------
    # At each step: apply BandMF to ALL rows, accumulate cold portion.
    # At each access step: deliver accumulated noise for those cold rows, zero them.
    solver_naive = CorNoiseGenerator(C, bandwidth=bandwidth, device='cpu', partition=1)

    accum_naive = torch.zeros(n_rows, m_spa)
    # delivered_naive[row] = list of (step, tensor(m_spa,))
    delivered_naive: dict[int, list[tuple[int, torch.Tensor]]] = {r: [] for r in cold_indices}

    for j in range(n_steps):
        z_j = z_all[j].clone()
        x_j = solver_naive.step(z_j)          # BandMF on ALL rows (1D, n_rows*m_spa)
        solver_naive.advance()
        x_j = x_j.view(n_rows, m_spa)
        accum_naive += x_j                    # accumulate ALL rows (cold portion included)

        for r in access_pattern[j]:
            if r in cold_set:
                delivered_naive[r].append((j, accum_naive[r].clone()))
                accum_naive[r].zero_()        # consumed — reset

    # Final step residual for unreached cold rows
    for r in cold_indices:
        if accum_naive[r].abs().sum().item() > 0:
            delivered_naive[r].append((n_steps, accum_naive[r].clone()))

    # -----------------------------------------------------------------------
    # Preprocessing simulation (NoiseCache-style)
    # -----------------------------------------------------------------------
    # At each step: SLICE z_full to the cold rows, apply a SEPARATE (cold-only) BandMF,
    # accumulate, deliver at access steps.
    solver_pre = CorNoiseGenerator(C, bandwidth=bandwidth, device='cpu', partition=1)

    accum_pre = torch.zeros(n_cold, m_spa)
    delivered_pre: dict[int, list[tuple[int, torch.Tensor]]] = {r: [] for r in cold_indices}

    for j in range(n_steps):
        z_j_full = z_all[j]
        # Slice EXACTLY the cold-row portion of z_full — same float values
        z_j_cold = z_j_full.view(n_rows, m_spa)[cold_indices].reshape(-1)   # (n_cold * m_spa,)

        x_j_cold = solver_pre.step(z_j_cold)  # BandMF on cold rows only
        solver_pre.advance()
        x_j_cold = x_j_cold.view(n_cold, m_spa)
        accum_pre += x_j_cold

        for r in access_pattern[j]:
            if r in cold_set:
                li = cold_local[r]
                delivered_pre[r].append((j, accum_pre[li].clone()))
                accum_pre[li].zero_()

    for r in cold_indices:
        li = cold_local[r]
        if accum_pre[li].abs().sum().item() > 0:
            delivered_pre[r].append((n_steps, accum_pre[li].clone()))

    # -----------------------------------------------------------------------
    # Compare: model-parameter-side accumulation, step by step
    #
    # Both approaches deliver noise in chunks at access steps.
    # model_naive[r] and model_pre[r] simulate the running total applied to
    # the GPU-side model parameter for each cold row — these must agree at
    # every step.
    # -----------------------------------------------------------------------

    # Index deliveries by step for fast lookup
    naive_at: dict[int, dict[int, torch.Tensor]] = {}
    for r in cold_indices:
        for s, v in delivered_naive[r]:
            naive_at.setdefault(s, {})[r] = v

    pre_at: dict[int, dict[int, torch.Tensor]] = {}
    for r in cold_indices:
        for s, v in delivered_pre[r]:
            pre_at.setdefault(s, {})[r] = v

    model_naive = torch.zeros(n_cold, m_spa)
    model_pre   = torch.zeros(n_cold, m_spa)

    print("Step-wise model side noise accumulation comparison (per delivery, cold rows only):")
    print(f"{'Step':>5}  {'Row':>4}  {'NaiveNorm':>10}  {'PreComputeNorm':>10}  {'MaxAbsErr':>12}")
    print("-" * 50)

    max_err_global = 0.0
    for j in range(n_steps + 1):
        naive_step = naive_at.get(j, {})
        pre_step   = pre_at.get(j, {})
        all_rows   = sorted(set(naive_step) | set(pre_step))

        for r, v in naive_step.items():
            model_naive[cold_local[r]] += v
        for r, v in pre_step.items():
            model_pre[cold_local[r]] += v

        for r in all_rows:
            li = cold_local[r]
            vn = model_naive[li]
            vp = model_pre[li]
            err = (vn - vp).abs().max().item()
            max_err_global = max(max_err_global, err)
            print(f"{j:>5}  {r:>4}  {vn.norm().item():>10.4f}  {vp.norm().item():>10.4f}  {err:>12.2e}")

    print("-" * 50)
    print(f"  Max abs err across all deliveries: {max_err_global:.2e}")
    print()

    # -----------------------------------------------------------------------
    # Buffer size comparison: naive vs preprocessing
    #
    # Naive real-time:  history buf = (bw-1) * n_rows * m_spa floats
    # Preprocessing:    hot history  = (bw-1) * n_hot  * m_spa floats
    #                   cold cache   = n_deliveries * m_spa floats
    #                                  (one precomputed chunk per (cold_row, access_step))
    # -----------------------------------------------------------------------
    n_hot        = len(hot_indices)
    bw           = bandwidth
    BYTES        = 4  # float32
    n_deliveries = sum(len(delivered_naive[r]) for r in cold_indices)

    PASS_THRESHOLD = 1e-5
    if max_err_global < PASS_THRESHOLD:
        print(f"PASSED  (max err {max_err_global:.2e} < {PASS_THRESHOLD:.0e})")
        print("Preprocessing noise == naive accumulated noise within 1e-5.")
    else:
        print(f"FAILED  (max err {max_err_global:.2e} >= {PASS_THRESHOLD:.0e})")
        sys.exit(1)

    naive_buf  = (bw - 1) * n_rows * m_spa
    hot_buf    = (bw - 1) * n_hot  * m_spa
    cold_cache = n_deliveries * m_spa
    pre_total  = hot_buf + cold_cache

    print("Buffer size comparison (float32):")
    print(f"  Naive  _buf  : (bw-1={bw-1}) × n_rows={n_rows} × m_spa={m_spa} = "
          f"{naive_buf:>8,} floats  ({naive_buf*BYTES/1024:.1f} KB)  [real-time noise history]")
    print(f"  Prepro hot   : (bw-1={bw-1}) × n_hot={n_hot}  × m_spa={m_spa} = "
          f"{hot_buf:>8,} floats  ({hot_buf*BYTES/1024:.1f} KB)  [real-time noise history]")
    print(f"  Prepro cold  :   n_deliveries={n_deliveries} × m_spa={m_spa} = "
          f"{cold_cache:>8,} floats  ({cold_cache*BYTES/1024:.1f} KB)  [precomputed cache]")
    print(f"  Prepro total :                                      "
          f"{pre_total:>8,} floats  ({pre_total*BYTES/1024:.1f} KB)")
    print(f"  MEM Reduction    : {naive_buf / pre_total:.2f}× smaller  "
          f"({100*(1 - pre_total/naive_buf):.1f}% saved)")
    print()


if __name__ == '__main__':
    main()
