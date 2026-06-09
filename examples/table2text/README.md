# table2text — LLM training with BandMF

DP fine-tuning of GPT-2 / OPT models on E2E and DART datasets,
using DeepSpeed ZeRO-1 with BandMF correlated noise and CPU partition offloading.

---

## Getting the data

E2E and DART datasets are adapted from [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) and hosted by [Li et al., 2021](https://arxiv.org/abs/2110.05679) at [Google Drive](https://drive.google.com/file/d/1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7/view?usp=sharing). To obtain the data, run:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7
unzip prefix-tuning.zip
```

This produces a `prefix-tuning/` folder containing both `e2e_data/` and `dart/`. Place it at `{workdir}/data/prefix-tuning/` so that the default `PREFIX_PATH` in the training scripts resolves correctly.

---

## Scripts

| Script | Purpose |
|---|---|
| `run_llm.sh` | Figure 18: end-to-end LLM timing — auto-profiles hardware, selects partitions, runs all models |
| `run_figure3.sh` | Figure 3: single-iteration time vs band size sweep (No partition) |
| `profile_hardware.sh` | Hardware profiler (called internally by `run_llm.sh`) |
| `run_ZERO1.sh` | DeepSpeed ZeRO-1 training wrapper (called by the above scripts) |
| `run_language_modeling.py` | HuggingFace training script; wires BandMF + DeepSpeed |
| `trainer.py` | Custom Trainer: DP step, CPU noise thread, CXL emulation, timing |
| `gpt_config_stage*.json` | DeepSpeed ZeRO-1 configs for different batch sizes |

---

## Partition configuration

```
PARTITION  = PART_GPU + PART_CPU          (total noise partitions)
PART_GPU   = 1                            (always; GPU handles one partition's GEMV)
PART_CPU   = N                            (CPU subprocess handles N partitions)
```

CPU `_buf` memory per rank: `PART_CPU × (band - 1) × (num_params / PARTITION / num_GPUs) × 4B`

Partition parameters are selected automatically by `run_llm.sh` based on hardware profiling — no manual tuning needed.

---

## Running

### Quick single run

```bash
bash run_ZERO1.sh \
  /path/to/prefix-tuning /path/to/output \
  e2e gpt2-large \
  32 1024 \
  no 64 \
  4 29500 \
  1 1 0
```

### Sweep over bands / batch sizes

To manually test a specific partition configuration across a range of bands and batch sizes, use `run_deepspeed_auto.sh`:

```bash
PARTITION=20 GPU_PART=1 CPU_PART=5 MODEL=gpt2-large \
BASE_MAX_STEPS=1024 BANDS="64 112" BATCH_SIZES="32" \
bash ../run_deepspeed_auto.sh
```

### Supported models

Pass the HuggingFace model name directly as the `model_name_or_path` argument:

| Model | HuggingFace name |
|---|---|
| GPT-2 variants | `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` |
| OPT-350M | `facebook/opt-350m` |
| OPT-1.3B | `facebook/opt-1.3b` |
| OPT-2.7B | `facebook/opt-2.7b` |
| OPT-6.7B | `facebook/opt-6.7b` |

---

## DP arguments

These are passed through `run_ZERO1.sh` to `run_language_modeling.py`:

| Argument | Default | Description |
|---|---|---|
| `task_mode` | — | Dataset: `e2e` or `dart` |
| `target_epsilon` | 8 | Privacy budget ε |
| `target_delta` | 8e-6 (E2E) / 1e-5 (DART) | Privacy budget δ |
| `per_example_max_grad_norm` | 0.1 | Per-sample gradient clipping norm *C* |
| `clipping_fn` | `automatic` | Per-sample clipping method: `automatic` ([Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)), `Abadi`, `global` |
| `clipping_mode` | `MixGhostClip` | DP algorithm: `MixOpt`, `MixGhostClip`, `ghost` ([Bu et al., 2022](https://arxiv.org/pdf/2210.00038.pdf)) |
| `non_private` | `no` | Set to `yes` to disable DP (baseline) |
| `min_separation` | 1 | BandMF bandwidth *b* (= band size) |

---

## BandMF / noise partition arguments

| Argument | Default | Description |
|---|---|---|
| `noise_partition` | 1 | Total partitions `P = PART_GPU + PART_CPU + PART_CXL (remainder)` |
| `GPU_partition` | 1 | Partitions handled on GPU (always 1) |
| `CPU_partition` | 0 | Partitions offloaded to CPU subprocess |
| `num_GPUs` | 8 | Number of GPUs for DeepSpeed ZeRO-1 |
| `max_steps` | 0 | Stop after N optimizer steps (0 = run full epochs) |
| `speed_mode` | `True` | Skip BandMF factorization; use a random banded matrix. For timing benchmarks only — noise is not privacy-correct. |

---

## Positional arguments to `run_ZERO1.sh`

| Position | Name | Default | Notes |
|---|---|---|---|
| 1 | `data_dir` | — | Path to `prefix-tuning/` root |
| 2 | `output_dir` | — | Output directory |
| 3 | `task_mode` | `e2e` | `e2e` or `dart` |
| 4 | `model_name_or_path` | `gpt2-medium` | model name |
| 5 | `physical_batch_size` | 40 | Per-GPU mini-batch size |
| 6 | `batch_size` | 1000 | Logical (global) batch size |
| 7 | `non_private` | `no` | `yes` to disable DP |
| 8 | `min_separation` | 1 | BandMF bandwidth *b* |
| 9 | `num_GPUs` | 8 | Number of GPUs |
| 10 | `masterport` | 61000 | DeepSpeed master port |
| 11 | `noise_partition` | 1 | Total `PARTITION` |
| 12 | `GPU_partition` | 1 | `PART_GPU` |
| 13 | `CPU_partition` | 0 | `PART_CPU` |
| 22 | `max_steps` | 0 | Stop after N steps (0 = use epochs) |
| 23 | `speed_mode` | `False` | `True` to skip BandMF factorization (timing benchmarks only) |

---

## CXL emulation

`trainer.py` contains an analytical CXL latency model (`cxl_noise_cxl`) that estimates the noise generation time when the GEMV history buffer resides on CXL-attached memory.
Throughput/Bandwidth parameters are fixed constants; no real CXL hardware is required.
