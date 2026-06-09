"""
Sanity check: BandMF CPU noise worker pipelining.

Verifies that CPU GEMV + H2D transfer completes within one simulated
training step, so the noise worker adds zero visible overhead to training.

A subprocess (identical to the real noise_worker code path) runs BandMF
GEMV and pushes noise tensors into a queue.  The main process simulates
training by sleeping for --step-ms, then consuming from the queue and
recording how long it had to wait.

  avg_wait ≈ 0 ms  →  PASS: worker is fully pipelined
  avg_wait > 0 ms  →  FAIL: worker is slower than training; increase PART_CPU or threads

Usage:
    # Scale --num-params to: model_params / n_gpus / noise_partition
    # Scale --step-ms to the measured step latency from profile_hardware.sh

    # Quick single-GPU test (no multi-GPU needed, fast to run):
    python sanity_noise_worker.py --num-params 6000000 --band-size 8 \\
        --partition 1 --step-ms 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastDP.bandmf import CorNoiseGenerator


def _worker(queue, stop_event, num_params, band_size, partition, C_np, threads):
    if threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)

    unit_size = num_params // partition
    gen = CorNoiseGenerator(C_np, bandwidth=band_size, device='cpu', partition=partition)
    gen.skip_to_steady_state()

    assert gen._matrix.device.type == 'cpu', f"matrix on wrong device: {gen._matrix.device}"
    _check = torch.normal(0.0, 1.0, (unit_size,))
    assert _check.device.type == 'cpu', f"noise tensor on wrong device: {_check.device}"
    print(f"[worker] device check OK — matrix={gen._matrix.device}  noise={_check.device}  "
          f"buf_size={(band_size-1)*unit_size*4/1e6:.1f} MB to read per GEMV", flush=True)

    gemv_times = []

    while not stop_event.is_set():
        for j in range(partition):
            raw = torch.normal(0.0, 1.0, (unit_size,))
            t0 = time.perf_counter()
            cor = gen.step(raw, j)
            gemv_times.append(time.perf_counter() - t0)
            cor_gpu = cor.to('cuda:0', non_blocking=True)
            while queue.full():
                if stop_event.is_set():
                    return
                time.sleep(0.001)
            queue.put(cor_gpu)
        gen.advance()

    if gemv_times:
        avg_ms = np.mean(gemv_times) * 1e3
        bw_gbs = ((band_size - 1) * unit_size * 4) / (avg_ms * 1e-3) / 1e9
        print(f"[worker] GEMV/partition: {avg_ms:.3f} ms  "
              f"BW: {bw_gbs:.2f} GB/s  "
              f"total/step ({partition} partitions): {avg_ms * partition:.1f} ms", flush=True)


def run(args):
    rng = np.random.default_rng(42)
    # Synthetic lower-triangular C matrix (enough rows for the run)
    n_rows = max(1000, args.steps + 10)
    C_np = np.tril(rng.standard_normal((n_rows, n_rows)).astype(np.float32))
    np.fill_diagonal(C_np, np.abs(np.diag(C_np)) + 0.5)

    mp.set_start_method('spawn', force=True)
    queue = mp.Queue(maxsize=args.partition)
    stop_event = mp.Event()

    p = mp.Process(
        target=_worker,
        args=(queue, stop_event, args.num_params, args.band_size,
              args.partition, C_np, args.threads),
    )
    p.start()
    # Simulating GPU-Side Training
    wait_times = []
    for i in range(args.steps):
        time.sleep(args.step_ms / 1000.0)
        t_wait = time.perf_counter()
        for _ in range(args.partition):
            queue.get()
        wait_ms = (time.perf_counter() - t_wait) * 1e3
        wait_times.append(wait_ms)
        if (i + 1) % 10 == 0 or args.steps <= 10:
            print(f"  step {i+1:3d}/{args.steps}  queue_wait={wait_ms:.2f} ms", flush=True)

    stop_event.set()
    p.join()

    avg_wait = float(np.mean(wait_times))
    max_wait = float(np.max(wait_times))
    overhead_pct = avg_wait / args.step_ms * 100

    threshold_ms = max(1.0, args.step_ms * 0.02)

    print(f"\n{'='*60}")
    print(f"  step_ms (simulated):  {args.step_ms:.1f} ms")
    print(f"  avg queue wait:       {avg_wait:.2f} ms  ({overhead_pct:.1f}% of step)")
    print(f"  max queue wait:       {max_wait:.2f} ms")
    if avg_wait <= threshold_ms:
        print(f"  RESULT: PASS  (< {threshold_ms:.1f} ms threshold; noise fully pipelined)")
        return 0
    else:
        print(f"  RESULT: FAIL  (avg {avg_wait:.1f} ms > {threshold_ms:.1f} ms threshold)")
        print(f"  Suggestion: increase --step-ms, reduce --band-size, "
              f"or add more OMP threads (--threads).")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check BandMF CPU noise worker pipelining.")
    parser.add_argument("--num-params", type=int, default=5_000_000,
                        help="Params per CPU partition unit "
                             "(= model_params / n_gpus / noise_partition)")
    parser.add_argument("--band-size",  type=int, default=64,
                        help="BandMF bandwidth (min_separation)")
    parser.add_argument("--partition",  type=int, default=5,
                        help="Number of CPU partitions per GPU (PART_CPU)")
    parser.add_argument("--step-ms",    type=float, default=3000.0,
                        help="Simulated training step latency (ms) — use profile_hardware.sh output")
    parser.add_argument("--steps",      type=int, default=30)
    parser.add_argument("--threads",    type=int, default=14,
                        help="OMP/MKL thread count for the CPU worker")
    args = parser.parse_args()

    print(f"num_params={args.num_params:,}  band={args.band_size}  "
          f"partition={args.partition}  step_ms={args.step_ms}  threads={args.threads}")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
