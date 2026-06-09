# Cocoon Artifact

**Cocoon: A System Architecture for Differentially Private Training with Correlated Noises**

[Paper (OSDI '26)](https://www.usenix.org/conference/osdi26/presentation/kim-donghwan) | [arXiv](https://arxiv.org/abs/2510.07304) | [Artifact (Zenodo)](https://doi.org/10.5281/zenodo.20422702)

---

Differentially private training with **Banded Matrix Factorization (BandMF)** correlated noise,
extending [fast-differential-privacy](https://github.com/awslabs/fast-differential-privacy).

Instead of i.i.d. Gaussian noise at every step, BandMF generates temporally correlated noise
whose covariance is optimised offline to minimise total or worst-case per-step variance.

---

## Repository Overview

| Folder | Description |
|---|---|
| `fastDP/` | Core library: PrivacyEngine, BandMF GEMV, CPU noise worker, DeepSpeed patch |
| `examples/` | Experiment scripts for LLM, DLRM, and image classification — see [`examples/README.md`](examples/README.md) |
| `benchmark/` | Correctness checks and GEMV throughput benchmarks — start with `run_getting_started.sh` |
| `pfl/` | BandMF matrix solvers (from Apple PFL-Research, unmodified) |
| `train_utils/` | Shared training utilities (data loaders, schedulers) |

---

## Getting Started Instructions

**Expected time: less than 30 minutes.**
These checks require only a single GPU and a Python environment with PyTorch.
They do not require the full training datasets.

### Step 0: Install

```bash
conda create -n cocoon python=3.10
conda activate cocoon
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -m setup develop
```

### Step 1: Run all getting-started checks

```bash
cd benchmark
bash run_getting_started.sh
```

Key overrides (set before running):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU indices to use (default: 4,5,6,7)
NUM_GPUS=4                     # number of GPUs
THREADS=14                     # CPU threads per noise worker
NUMA_CPUS="21-27,77-83"        # CPU cores for single-worker checks (match your NUMA topology)
NUMA_CPUS_MULTI="21-27,77-83:28-34,84-90:35-41,91-97:42-48,98-104"  # per-GPU cores for multi-GPU check
BAND=32                        # BandMF bandwidth
```

This runs four checks in sequence:

**[1/4] BandMF GEMV correctness** — unit test for `CorNoiseGenerator.step()`.
Verifies the output matches the explicit forward-substitution recurrence and
that CPU→GPU transfer is correct.

**[2/4] CPU/GPU GEMV throughput** — measures raw GEMV throughput.
The key number is GEMV ms/step: must be less than the training step latency for
full pipelining. Sub-checks:

- [2a] Single-partition CPU GEMV (1 subprocess worker, pinned to `NUMA_CPUS`)
- [2b] Multi-GPU CPU GEMV (N concurrent workers, one per GPU; skipped if `NUM_GPUS=1`)
- [2c] GPU-native GEMV (all on CUDA, no H2D; baseline comparison)

**[3/4] Noise worker pipeline check** — simulates a training loop with `time.sleep`
(no model loaded) and measures how long the main process waits for noise.
A near-zero wait means GEMV is fully hidden behind training.

**[4/4] Noise preprocessing correctness** — verifies that the DLRM `NoiseCache`
coalescing produces the same per-row noise as a naive per-row BandMF simulation.

---

## Detailed Instructions

### Hardware setup

All paper experiments were run on:

| Component | Setup A (primary) | Setup B |
|---|---|---|
| GPU | 4 × NVIDIA RTX A5000 24 GB (CUDA 12.2) | 4 × NVIDIA A100 80 GB (CUDA 11.8) |
| CPU | Intel Xeon Gold 6330 (28 cores, HT) | AMD EPYC 7763 (64 cores) |
| RAM | 128 GB | 512 GB |
| Interconnect | PCIe 4.0 x16 per GPU | PCIe 4.0 x16 per GPU |

### NUMA configuration

All run scripts pin processes to the NUMA node that is physically closest to the target GPUs
using `numactl`. This minimises PCIe latency for CPU↔GPU noise transfers.
Install `numactl` if not already present:

```bash
sudo apt-get install numactl       # Debian/Ubuntu
sudo yum install numactl           # RHEL/CentOS
```

**Step 1 — Find which NUMA node each GPU is on:**

```bash
nvidia-smi topo -m
```

Look for the `CPU Affinity` and `NUMA Affinity` columns. Example output:

```
        GPU0  GPU1  GPU2  GPU3  CPU Affinity  NUMA Affinity
GPU0     X    PXB   PXB   PXB   0-27,56-83  1
GPU1    PXB    X    PXB   PXB   0-27,56-83  1
GPU2    PXB   PXB    X    PXB   0-27,56-83  1
GPU3    PXB   PXB   PXB    X    0-27,56-83  1
```

**Step 2 — Verify NUMA topology:**

```bash
numactl --hardware
```

This shows NUMA nodes, their CPU ranges, and available memory. Example:

```
available: 2 nodes (0-1)
node 0 cpus: 0-27 56-83
node 0 size: 128602 MB
node 1 cpus: 28-55 84-111
node 1 size: 128955 MB
```

**Step 3 — Fill in the variables in each run script:**

| Variable | Meaning | Example |
|---|---|---|
| `NUMA_CPUS` | CPU core range for the GPU's NUMA node | `28-55,84-111` |
| `NUMA_MEM` | NUMA memory node index | `1` |

These variables appear near the top of each run script (e.g. `run_dlrm.sh`,
`run_figure5.sh`, `run_deepspeed_auto.sh`)

### Experiments by script

| Script | Location | Figures | What it measures |
|---|---|---|---|
| `run_figure3.sh` | [`examples/table2text/`](examples/table2text/) | Figure 3 | Single-iteration time vs band size for LLM (OPT, GPT-2) with ZeRO-1; GPU memory limit sweep |
| `run_figure5.sh` | [`examples/image_classification/`](examples/image_classification/) | Figure 5 | ViT-Large step breakdown: DP-SGD vs GPU-GEMV vs CPU-GEMV for 1/2/4 GPUs |
| `run_dlrm.sh` | [`examples/dlrm/`](examples/dlrm/) | Figures 14–15 | DLRM end-to-end: embedding table noise breakdown, scalability, and efficiency |
| `run_llm.sh` | [`examples/table2text/`](examples/table2text/) | Figure 18 | LLM end-to-end normalized training time: Cocoon+NMP vs GPU-GEMV vs CPU-GEMV |

---

## Attribution

BandMF noise mechanism from:
> *(Amplified) Banded Matrix Factorization: A unified approach to private training*
> https://arxiv.org/pdf/2306.08153.pdf

Matrix factorization solvers from
[Apple PFL-Research](https://github.com/apple/pfl-research) (Apache 2.0).

DP clipping and per-sample gradient sampling from
[fast-differential-privacy](https://github.com/awslabs/fast-differential-privacy) (Apache 2.0).

DLRM model from [facebookresearch/dlrm](https://github.com/facebookresearch/dlrm) (MIT).
