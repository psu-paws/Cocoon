# image_classification — ViT-Large DeepSpeed ZeRO benchmarks

Benchmarks for Figure 5: single-iteration time breakdown of DP-SGD vs GPU-GEMV vs CPU-GEMV
for ViT-Large with 1, 2, and 4 GPUs, using DeepSpeed ZeRO stage 1.

---

## Scripts

| Script | Purpose |
|---|---|
| `CIFAR_TIMM_ZERO1.py` | Training script (FakeData, ViT-Large, ZeRO-1, BandMF noise offload) |
| `run_figure5.sh` | Figure 5 benchmark: runs DP-SGD / CPU-GEMV / GPU-GEMV for 1/2/4 GPUs |

Config files: `cifar_config.json` (batch=1024, micro-batch=64, grad_accum=16 per GPU).

---

## Figure 5: Single-iteration time breakdown

```bash
bash run_figure5.sh          # all GPU counts (1, 2, 4)
bash run_figure5.sh 1        # single GPU count
```

**What it measures:** Per-step wall-clock time broken down by component for three noise modes:

| Mode | Description |
|---|---|
| DP-SGD | Standard i.i.d. Gaussian noise, band=1 |
| GPU-GEMV | BandMF GEMV kept on GPU; overhead measured analytically (GEMV time + PCIe transfer for history overflow) |
| CPU-GEMV | BandMF GEMV offloaded to CPU subprocess (`--noise_offload`); GEMV fully hidden behind GPU training |

**Parameters (edit in `run_figure5.sh`):**

| Variable | Default | Meaning |
|---|---|---|
| `TARGET_BAND` | 16 | BandMF bandwidth for CPU-GEMV and GPU-GEMV modes |
| `NUM_BATCHES` | 10 | Logical steps per run (warmup=5 excluded from timing) |
| `PCIE_BW_GBS` | 32.0 | PCIe bandwidth used for GPU-GEMV analytical transfer estimate |
| `GPU_DEVICES` | `[1]=4 [2]=4,5 [4]=4,5,6,7` | CUDA device indices per GPU count |
| `NUMA_CPUS` | see script | numactl CPU binding per GPU count |

**Phases:**

1. **DP-SGD** (band=1): measures baseline training step time (commented out by default; re-enable if needed)
2. **CPU-GEMV** (`--noise_offload --noise-num-threads N`): measures training step with GEMV hidden behind GPU compute; reports `Avg_Iter` ms
3. **GPU-GEMV** (analytical): binary-searches for the largest band that fits in GPU memory without OOM, benchmarks GEMV with `bench_gemv_gpu.py`, scales to `TARGET_BAND` analytically

**Output:** Summary table printed at end, per-run logs saved to `figure5/`.

---

## Timing breakdown (printed per run)

```
step_by_step    : Xms   ← wall-clock logical step time (step-to-step)
iter            : Xms   ← forward+loss+backward × grad_accum
forward         : Xms   ← per micro-batch
loss            : Xms   ← per micro-batch
backward        : Xms   ← per micro-batch, excluding clip
clip            : Xms   ← per-sample gradient clipping
optim_step      : Xms   ← optimizer step (ZeRO allreduce + weight update)
ngen            : Xms   ← GPU-side noise generation (0 when CPU-GEMV with part_GPU=0)
nadd            : Xms   ← noise addition to gradient partition
h2d_transfer    : Xms   ← CPU→GPU transfer benchmark
```

---

## Key flags for `CIFAR_TIMM_ZERO1.py`

| Flag | Meaning |
|---|---|
| `--min_separation B` | BandMF bandwidth `b` (1 = standard DP-SGD) |
| `--noise_offload` | Enable CPU subprocess noise worker |
| `--noise-num-threads N` | OMP/MKL thread count for the noise worker subprocess |
| `--num_batches N` | Stop after N logical steps per epoch (0 = full epoch) |
| `--model` | TIMM model name (default: `vit_large_patch16_224`) |
