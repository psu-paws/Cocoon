# Cocoon Examples

Experiment scripts for differentially private training with BandMF correlated noise.
See the [root README](../README.md) for installation, getting-started checks, and paper context.

---

## Running experiments

| Script | Figures | Command |
|---|---|---|
| `table2text/run_llm.sh` | Figure 18 | `bash examples/table2text/run_llm.sh` |
| `dlrm/run_dlrm.sh` | Figures 14–17 | `bash examples/dlrm/run_dlrm.sh` |
| `image_classification/run_figure5.sh` | Figure 5 | `bash examples/image_classification/run_figure5.sh` |

All scripts read data from `{workdir}/data/` by default (override with `DATA_DIR=...`).
Partition parameters (`n_part`, `G`, `C`, `NMP`) are selected automatically based on
hardware profiling — no manual tuning needed.

---

## Repository layout

```
examples/
├── run_deepspeed_auto.sh        # manual sweep runner: band × batch size grid
│
├── table2text/
│   ├── run_llm.sh               # Figure 18: end-to-end LLM timing (main entry point)
│   ├── profile_hardware.sh      # hardware profiler (called internally by run_llm.sh)
│   ├── run_ZERO1.sh             # DeepSpeed ZeRO-1 LLM training wrapper
│   ├── run_language_modeling.py # HuggingFace training script + DP patches
│   └── trainer.py               # custom Trainer with BandMF + CPU offload
│
├── dlrm/
│   ├── run_dlrm.sh              # Figures 14–15: DLRM timing breakdown (main entry point)
│   ├── dlrm_s_pytorch.py        # DLRM training with BandMF + preprocessing
│   └── dlrm_preprocessing.py   # hot/cold row partition + noise cache builder
│
└── image_classification/
    ├── run_figure5.sh           # Figure 5: ViT-Large DP-SGD vs GEMV, 1/2/4 GPUs
    └── CIFAR_TIMM_ZERO1.py      # ViT training with BandMF + CPU offload
```

---

## Hardware requirements

| Component | Spec |
|---|---|
| GPU | 4 × NVIDIA A5000 24 GB (CUDA 12.2) |
| CPU | Intel Xeon Gold 6330 (56 cores, HyperThreaded) |
| RAM | 128 GB per NUMA node |
| Interconnect | PCIe 4.0 per GPU |
