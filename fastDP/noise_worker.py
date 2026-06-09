"""CPU subprocess worker for BandMF correlated noise offloading.

Shared by all training scripts (image classification, DLRM, LLM trainer).
Generates per-step BandMF-correlated noise on CPU and pushes to a queue
consumed by the main GPU training process.

Two modes (selected by raw_noise_queue):
  CPU-gen  (raw_noise_queue=None): sample z ~ N(0, sigma^2 I) on CPU, apply BandMF GEMV. (Not recommended, CPU Noise Gen too slow)
  GPU-feed (raw_noise_queue set ): receive raw Gaussian from GPU via raw_noise_queue, apply GEMV on CPU.
"""
import time
import os
import torch


def noise_worker(
    noise_queue,
    stop_event,
    num_params,
    noise_multiplier,
    bandmf_solver,
    raw_noise_queue=None,
    partition=1,
    transfer_to_gpu=True,
    gpu_device='cuda:0',
    dev_rank=0,
    num_threads=None,
    verbose=True,
    speed_mode=False,
):
    """CPU subprocess: generate BandMF-correlated noise each step and push to noise_queue.

    Args:
        noise_queue: output Queue; receives one correlated noise tensor per partition per step.
        stop_event: multiprocessing.Event; worker exits when set or all iterations exhausted.
        num_params: flat parameter count; used for Gaussian sampling in CPU-gen mode.
        noise_multiplier: Gaussian std (= sigma * max_grad_norm).
        bandmf_solver: BandedMatrixFactorizationMechanism or CorNoiseGenerator.
        raw_noise_queue: if set, receive raw Gaussian from GPU via this queue instead of generating.
        partition: number of noise partitions per step for BandMF step().
        transfer_to_gpu: if True, pin_memory() + H2D transfer before putting to noise_queue.
            Set False when the consumer (main process) handles H2D itself.
        gpu_device: target GPU device string for H2D transfer (e.g. 'cuda:0').
        dev_rank: process rank; added to seed for independent randomness across workers.
        num_threads: if set, overrides OMP/MKL/NUMEXPR thread counts for this subprocess.
        verbose: if True, print per-step timing.
    """
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
    torch.manual_seed(42 + dev_rank)
    if num_threads is not None:
        os.environ["OMP_NUM_THREADS"]       = str(num_threads)
        os.environ["MKL_NUM_THREADS"]       = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"]  = str(num_threads)
        os.environ["NUMEXPR_NUM_THREADS"]   = str(num_threads)
        torch.set_num_threads(num_threads)
        # OS-level CPU pinning: partition the inherited affinity set by dev_rank so each
        # worker is restricted to its own non-overlapping core slice.
        try:
            all_cpus = sorted(os.sched_getaffinity(0))
            chunk = min(num_threads, len(all_cpus))
            start = (dev_rank * chunk) % len(all_cpus)
            my_cpus = set(all_cpus[start : start + chunk])
            os.sched_setaffinity(0, my_cpus)
        except (AttributeError, OSError):
            pass  # sched_setaffinity not available on this platform
    print(f"[noise_worker rank={dev_rank}] threads={num_threads} "
          f"affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'N/A'}", flush=True)

    gemv_times = []
    transfer_times = []
    total_times = []

    num_iterations = getattr(bandmf_solver, 'num_iterations', None)
    unit_size = num_params // partition
    _bw = getattr(getattr(bandmf_solver, '_solver', bandmf_solver), '_bandwidth', None)
    _chunk_size = None  # learned from first tensor in GPU-feed mode

    print(f"[DEBUG noise_worker] started, partition={partition} num_iterations={num_iterations}", flush=True)

    # In speed_mode: pre-generate one raw noise vector and reuse every step.
    # Measures GEMV + transfer only, not per-step noise generation.
    fixed_raw_noise = None
    if speed_mode:
        fixed_raw_noise = torch.normal(0, noise_multiplier, (unit_size,), device='cpu')

    while not stop_event.is_set():
        # Exit when all BandMF steps are consumed (only defined on BandedMatrixFactorizationMechanism).
        current_step = (bandmf_solver.current_step
                        if hasattr(bandmf_solver, 'current_step')
                        else bandmf_solver._step)
        if num_iterations is not None and current_step >= num_iterations:
            break

        t0 = None  # set after first raw_noise arrives (GPU-feed) or on first generation (CPU-gen)

        # --- Interleaved GEMV + enqueue: get raw, compute, push result before fetching next ---
        # Must interleave to avoid deadlock: cpu_noise_cpu blocks on noise_queue.get() after
        # each put, so the result must be enqueued before requesting the next raw_noise.
        t_transfer_total = 0.0
        for j in range(partition):
            if fixed_raw_noise is not None:
                if t0 is None:
                    t0 = time.perf_counter()
                raw_noise = fixed_raw_noise
            elif raw_noise_queue is not None:
                raw_noise = raw_noise_queue.get()          # GPU-feed: receive from main process
                if t0 is None:
                    t0 = time.perf_counter()               # start timing once work actually begins
            else:
                if t0 is None:
                    t0 = time.perf_counter()
                raw_noise = fixed_raw_noise if fixed_raw_noise is not None \
                    else torch.normal(0, noise_multiplier, (unit_size,), device='cpu')
            if _chunk_size is None:
                _chunk_size = raw_noise.numel()
            t_gemv = time.perf_counter()
            cor_noise = bandmf_solver.step(raw_noise, j)
            gemv_times.append(time.perf_counter() - t_gemv)

            t_transfer = time.perf_counter()  
            while noise_queue.full():                     
                time.sleep(0.01)
            if transfer_to_gpu:
                cor_noise = cor_noise.to(gpu_device, non_blocking=True)
            noise_queue.put(cor_noise)
            t_transfer_total += time.perf_counter() - t_transfer # Since transfer is non-blocking, time can be inaccurate.

        bandmf_solver.advance()

        transfer_times.append(t_transfer_total)
        total_times.append(time.perf_counter() - t0)

        if verbose and num_iterations is not None and current_step == num_iterations - 1:
            n = len(total_times)
            avg_gemv_per_part = sum(gemv_times) / len(gemv_times) * 1000   # ms per partition
            avg_xfer_per_part = sum(transfer_times) / n * 1000 / partition  # ms per partition (queue wait + put)
            avg_total = sum(total_times) / n * 1000                         # ms per step (actual work only)
            if _chunk_size and _bw and gemv_times:
                _gemv_bytes = (_bw - 1) * _chunk_size * 4
                avg_bw_gbs = round(_gemv_bytes / (avg_gemv_per_part * 1e-3 * 1e9), 2)
            else:
                avg_bw_gbs = 0
            print(f"[noise_worker rank={dev_rank}] FINAL: CPU×{partition} steps={n} | "
                  f"gemv/part={avg_gemv_per_part:.2f}ms({avg_bw_gbs}GB/s) "
                  f"queue/part={avg_xfer_per_part:.2f}ms "
                  f"total/step={avg_total:.2f}ms (≈{avg_gemv_per_part*partition:.0f}ms gemv + {avg_xfer_per_part*partition:.0f}ms queue)", flush=True)

    # Keep the process alive until explicitly stopped.
    # Exiting immediately after the last noise_queue.put()
    print(f"[noise_worker] all steps done, waiting for stop_event", flush=True)
    while not stop_event.is_set():
        time.sleep(0.1)
