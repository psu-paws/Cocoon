# dlrm — DLRM training with BandMF

DP training of the Deep Learning Recommendation Model with BandMF correlated noise.
Single-GPU; embedding table noise uses a **hot/cold row partition** with a precomputed
noise cache — our contribution for handling the sparse access pattern of DLRM embeddings.

---

## Getting the data

**Primary (real Criteo Kaggle):**

```bash
wget https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
tar -xzf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
# → produces train.txt (set as --raw-data-file) and test.txt
```

The first training run converts `train.txt` → `kaggleAdDisplayChallenge_processed.npz` automatically (one-time, slow). Subsequent runs reuse the cached `.npz`.

**Recommended: resave uncompressed for fast loading.**
The default `.npz` uses gzip, which adds ~30–60 s per run. Resave once:

```python
import numpy as np
d = np.load("kaggleAdDisplayChallenge_processed.npz")
np.savez("kaggleAdDisplayChallenge_processed_fast.npz", **dict(d))
```

Point `--processed-data-file` at the `_fast.npz` file for all subsequent runs.

**Synthetic datasets** for distribution sensitivity studies (Figures 15b, 15d):

```bash
KAGGLE_INPUT=/path/to/kaggleAdDisplayChallenge_processed.npz \
DATA_DIR=/path/to/data \
bash ../../benchmark/SyntheData/generate_datasets.sh
```

This generates all five synthetic variants (Zipf α ∈ {0.5, 1.0, 2.0}, entry scale ∈ {0.5, 1.0, 2.0}) in one pass, skipping any that already exist.

---

## Our contribution: embedding noise preprocessing (Cocoon)

Standard BandMF generates noise for all parameters each step. For DLRM, embedding
tables are huge (33M+ rows) and most rows are not accessed in any given step.
We split rows into:

- **Hot rows** (accessed > `--thresholds` times per epoch): online BandMF noise generated each step by `PrivacyEngine`
- **Cold rows** (rarely accessed): noise precomputed offline by `NoiseCache` for the full training horizon and inserted per-step via indexed scatter

`dlrm_preprocessing.py` builds the `NoiseCache` by:
1. Scanning the full dataset to classify hot/cold rows by access frequency
2. Building per-step access-offset tensors (which cold rows are touched at each step)
3. Simulating the BandMF for `nepochs × steps_per_epoch` total steps per chunk
4. Storing per-step noise slices on CPU; transferred to GPU at training time

This converts `n_cold_rows × m_spa` online GEMV into a one-time coalesced offline cost, leaving only `n_hot_rows × m_spa` for online noise generation.

---

## Scripts

| File | Purpose |
|---|---|
| `run_dlrm.sh` | Reproduce Figures 14–15: timing breakdown, scalability, memory footprint |
| `dlrm_s_pytorch.py` | Main training script: BandMF noise + hot/cold preprocessing |
| `dlrm_preprocessing.py` | `NoiseCache` builder: hot/cold classification, offline BandMF simulation |
| `dlrm_data_pytorch.py` | Data loading (adapted from facebookresearch/dlrm) |

---

## Running

**Edit the user-adjustable section at the top of `run_dlrm.sh`** before running:

```bash
GPU=3                          # GPU index
NUMA_CPUS="21-27,77-83"        # CPU cores on the same NUMA node as the GPU
NUMA_MEM=0                     # NUMA memory node index
KAGGLE_PROC=/path/to/kaggleAdDisplayChallenge_processed_fast.npz
```
then run

```bash
bash run_dlrm.sh        # all figures
bash run_dlrm.sh 14     # only Figure 14
```

**Direct invocation** (single run):

```bash
python dlrm_s_pytorch.py \
  --processed-data-file /path/to/kaggleAdDisplayChallenge_processed_fast.npz \
  --raw-data-file /path/to/train.txt \
  --batch-size 65536 --mini-batch-size 65536 \
  --target-epsilon 3.0 --target-delta 2.55e-8 --max_grad_norm 30 \
  --min-separation 8 \
  --preprocessing --chunk-size 5000000 \
  --thresholds 5 \
  --nepochs 3 \
  --enable-profiling --speed-mode
```

Key arguments:

| Argument | Meaning |
|---|---|
| `--min-separation` | BandMF bandwidth *b* (1 = standard DP-SGD) |
| `--preprocessing` | Enable hot/cold partition + `NoiseCache` |
| `--thresholds` | Access count threshold to classify a row as hot |
| `--chunk-size` | Cold rows per `NoiseCache` chunk (tune to fit GPU memory during preprocessing) |
| `--nepochs` | Training epochs; noise is precomputed for the full `nepochs × steps_per_epoch` horizon |
| `--speed-mode` | Skip BandMF factorization (use a dummy banded matrix); for timing benchmarks only |
| `--preprocess-only` | Exit after `NoiseCache.build()` and print memory layout |

---

---

## Timing output (with `--enable-profiling`)

```
Avg_Iter Xms  Avg_Grad_Clip Xms  Avg_Noise_Gen Xms  Avg_Noise_Add Xms
Avg_Noise_Assemble Xms  Avg_Noise_Transfer Xms
```

| Field | Interval | What it covers |
|---|---|---|
| `Avg_Iter` | step-to-step wall clock | Full iteration |
| `Avg_Noise_Gen` | `start_noise_gen → end_noise_gen` | Online GEMV (hot rows + dense) + cold transfer |
| `Avg_Noise_Assemble` | `start_noise_gen → PPtransfer` | Hot row noise assembly on GPU |
| `Avg_Noise_Transfer` | `PPtransfer → end_noise_gen` | Cold row CPU → GPU scatter |
| `Avg_Noise_Add` | `end_noise_gen → end_noise_add` | Noise addition to gradient |
