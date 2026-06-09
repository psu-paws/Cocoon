"""
Benchmark: CPU-subprocess BandMF noise generation + H2D transfer.

Spawns one noise subprocess per GPU (default: 1). Each subprocess runs BandMF GEMV
on CPU and pushes the result to its GPU via a queue. The main process measures wait
latency per step.

Usage:
    python bench_gemv_cpu.py <num_params> <num_threads_per_worker> <bandwidth> [n_gpus=1] [cpu_affinities]
Example (single GPU):
    python bench_gemv_cpu.py 6000000 14 32
Example (4 GPUs, per-GPU CPU pinning):
    python bench_gemv_cpu.py 6000000 14 32 4 "28-34,84-90:35-41,91-97:42-48,98-104:49-55,105-111"

cpu_affinities: colon-separated CPU specs, one per GPU worker.
    Each spec is a comma-separated list of ranges, e.g. "28-34,84-90".
    When provided, each worker pins itself to its dedicated CPUs via sched_setaffinity
    and sets torch.set_num_threads to the number of pinned cores.
"""
import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastDP.bandmf import CorNoiseGenerator


def parse_cpu_spec(spec):
    """Parse "28-34,84-90" into a frozenset of CPU ids."""
    cpus = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = map(int, part.split('-'))
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


def noise_worker(noise_queue, stop_event, tensor_size, band_size, C_np, gpu_id, cpu_affinity=None):
    """Generate BandMF-correlated noise on CPU; push GPU tensors to queue."""
    # Pin this worker to its dedicated CPUs before any compute.
    if cpu_affinity is not None:
        try:
            os.sched_setaffinity(0, cpu_affinity)
        except OSError:
            pass  # non-Linux or insufficient privilege
        torch.set_num_threads(len(cpu_affinity))

    forward = CorNoiseGenerator(C_np, bandwidth=band_size, device='cpu')
    forward.skip_to_steady_state()

    noise = torch.normal(mean=0.0, std=1.0, size=(tensor_size,))
    run_gemv = []

    while not stop_event.is_set():
        t0 = time.perf_counter()
        cor_noise = forward.step(noise)
        forward.advance()
        t1 = time.perf_counter()
        run_gemv.append(t1 - t0)

        while noise_queue.full():
            if stop_event.is_set():
                return
            time.sleep(0.001)

        cor_noise.to(f'cuda:{gpu_id}', non_blocking=True)
        noise_queue.put(cor_noise)

    print(f"[GPU {gpu_id}] GEMV {np.mean(run_gemv)*1e3:.3f} ms")


def run_benchmark(tensor_size, iterations, band_size, n_gpus, cpu_affinities=None):
    rng = np.random.default_rng(42)
    C_np = np.tril(rng.standard_normal((1000, 1000)).astype(np.float32))
    np.fill_diagonal(C_np, np.abs(np.diag(C_np)) + 0.5)

    part_size = (tensor_size + n_gpus - 1) // n_gpus
    print(f"  n_gpus={n_gpus}  part_size={part_size:,} per GPU")
    if cpu_affinities:
        for i, aff in enumerate(cpu_affinities):
            print(f"  GPU[{i}] pinned to {len(aff)} CPUs: {sorted(aff)}")

    torch.multiprocessing.set_start_method('spawn', force=True)
    queues     = [torch.multiprocessing.Queue(maxsize=1) for _ in range(n_gpus)]
    stop_event = torch.multiprocessing.Event()

    processes = []
    for gpu_id in range(n_gpus):
        aff = cpu_affinities[gpu_id] if cpu_affinities else None
        p = torch.multiprocessing.Process(
            target=noise_worker,
            args=(queues[gpu_id], stop_event, part_size, band_size, C_np, gpu_id, aff),
        )
        p.start()
        processes.append(p)

    wall_times  = []
    cuda_starts = []
    cuda_ends   = []

    for i in range(iterations):
        cuda_starts.append(torch.cuda.Event(enable_timing=True))
        cuda_starts[-1].record()
        t0 = time.perf_counter()
        for q in queues:
            _ = q.get()
        t1 = time.perf_counter()
        cuda_ends.append(torch.cuda.Event(enable_timing=True))
        cuda_ends[-1].record()
        wall_times.append(t1 - t0)
        print(f"  iter {i+1}/{iterations}  wait {(t1-t0)*1e3:.3f} ms", flush=True)

    torch.cuda.synchronize()
    cuda_ms = [s.elapsed_time(e) for s, e in zip(cuda_starts, cuda_ends)]

    stop_event.set()
    for p in processes:
        p.join()

    avg_wall = np.mean(wall_times) * 1e3
    avg_cuda = np.mean(cuda_ms)
    print(f"\nWall-time  avg {avg_wall:.3f} ms  std {np.std(wall_times)*1e3:.3f} ms")
    print(f"CUDA-time  avg {avg_cuda:.3f} ms")
    print(f"(overhead = wall - GEMV; includes queue, CUDA IPC, scheduling)")
    return wall_times


def configure_threading(num_threads):
    os.environ["OMP_NUM_THREADS"]     = str(num_threads)
    os.environ["MKL_NUM_THREADS"]     = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def check_blas():
    cfg = torch.__config__.show()

    is_amd = False
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        is_amd = "AuthenticAMD" in cpuinfo
    except OSError:
        pass

    has_mkl      = "blas_info=mkl"      in cfg.lower() or "USE_MKL=ON"      in cfg
    has_openblas = "blas_info=openblas" in cfg.lower() or "USE_OPENBLAS=ON" in cfg

    if is_amd and has_mkl and not has_openblas:
        print("=" * 70)
        print("WARNING: AMD CPU detected with MKL BLAS.")
        print("  MKL often runs single-threaded on AMD CPUs, making GEMV extremely slow.")
        print("  Expected symptom: GEMV ms/step is ~num_threads× too slow.")
        print()
        print("  Fix: build PyTorch 2.4.0 from source with OpenBLAS:")
        print()
        print("    git clone --recursive https://github.com/pytorch/pytorch")
        print("    cd pytorch && git checkout v2.4.0")
        print("    git submodule sync && git submodule update --init --recursive")
        print("    export BLAS=OpenBLAS   # forces OpenBLAS")
        print("    export USE_MKL=0       # disables MKL (better for AMD EPYC)")
        print("    python setup.py develop")
        print("=" * 70)
    elif has_mkl:
        print("BLAS: MKL  (Intel CPU — expected good multi-threaded GEMV performance)")
        print("  If not installed with CUDA support, reinstall with:")
        print("    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "
              "--index-url https://download.pytorch.org/whl/cu124")
    elif has_openblas:
        print("BLAS: OpenBLAS  (good multi-threaded GEMV performance)")
    else:
        print("BLAS: unknown — check torch.__config__.show() output above")


def main():
    tensor_size      = int(sys.argv[1])
    num_threads      = int(sys.argv[2])   # per-worker thread count
    band_size        = int(sys.argv[3])
    n_gpus           = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    iterations       = 100

    # Optional: colon-separated per-GPU CPU specs, e.g. "28-34,84-90:35-41,91-97"
    cpu_affinities = None
    if len(sys.argv) > 5:
        specs = sys.argv[5].split(':')
        if len(specs) == n_gpus:
            cpu_affinities = [parse_cpu_spec(s) for s in specs]
        else:
            print(f"[warn] cpu_affinities has {len(specs)} specs but n_gpus={n_gpus}; ignoring")

    print(f"tensor_size={tensor_size:,}  num_threads_per_worker={num_threads}  "
          f"bandwidth={band_size}  n_gpus={n_gpus}")
    check_blas()
    configure_threading(num_threads)
    run_benchmark(tensor_size, iterations, band_size, n_gpus, cpu_affinities)


if __name__ == "__main__":
    main()
