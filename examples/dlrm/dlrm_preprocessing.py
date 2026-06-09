import torch
from fastDP.bandmf import BandedMatrixFactorizationMechanism
from collections.abc import Iterable
import numpy as np

def get_memory_usage(obj, seen_ids=None):
    """Recursively sum memory of a nested Python object (tensors, dicts, iterables) in bytes.

    Uses seen_ids to avoid double-counting shared references and circular structures.
    """
    if seen_ids is None:
        seen_ids = set()

    obj_id = id(obj)
    if obj_id in seen_ids:
        return 0

    seen_ids.add(obj_id)
    size = 0

    if isinstance(obj, torch.Tensor):
        if obj.storage():
            size += obj.storage().nbytes()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            size += get_memory_usage(k, seen_ids)
            size += get_memory_usage(v, seen_ids)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for item in obj:
            size += get_memory_usage(item, seen_ids)

    return size

def show_memory():
    """Print current, reserved, and peak GPU memory usage for device 0."""
    print("torch.cuda.memory_allocated: %fMB"%(torch.cuda.memory_allocated(0)/1024/1024))
    print("torch.cuda.memory_reserved: %fMB"%(torch.cuda.memory_reserved(0)/1024/1024))
    print("torch.cuda.max_memory_reserved: %fMB"%(torch.cuda.max_memory_reserved(0)/1024/1024))

def unpack_batch(b):
    """Unpack a dataloader batch; returns (X_int, lS_o, lS_i, T, ones, None)."""
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

def summarize_frequency_distribution(counts):
    """Bucket a per-row access count tensor into human-readable frequency ranges.

    Returns an ordered dict with exact counts 0-10, logarithmic range buckets, and a '>10M' tail.
    """
    summary = {}

    for i in range(11):
        n = (counts == i).sum().item()
        summary[f"Count of {i}"] = n

    bins = [100, 1000, 10000, 100000, 1000000, 10000000]
    lower_bound = 10
    for upper_bound in bins:
        n = ((counts > lower_bound) & (counts <= upper_bound)).sum().item()
        summary[f"{lower_bound + 1} - {upper_bound}"] = n
        lower_bound = upper_bound

    summary[f"> {lower_bound}"] = (counts > lower_bound).sum().item()

    return summary


class NoiseCache:
    """Pre-compute BandMF-correlated noise for cold (low-frequency) embedding rows.

    At training time, hot rows (accessed > threshold times per epoch) get fresh per-step noise
    generated on-the-fly by PrivacyEngine. Cold rows are rarely updated, so their accumulated
    BandMF noise is simulated once before training and stored here.

    Key data structures produced by build():
      - cold_chunks    : tuple of 1-D tensors, each covering one chunk of cold row IDs
      - access_offsets : access_offsets[step][chunk] — which cold rows are accessed at this step
      - noise_slices   : noise_slices[chunk][step]   — accumulated BandMF noise to add at step
      - residual_noise : residual_noise[chunk]        — leftover accumulation for the final step

    Parameter layout assumed for DLRM (26 sparse tables):
      [0 .. n_emb_params)       — embedding parameters (n_emb_rows × m_spa)
      [n_emb_params .. total)   — dense MLP parameters
    """

    def __init__(self, dlrm, train_ld, train_data, bandmf_solver, thresholds,
                 ln_emb, m_spa, epochs, noise_std, chunk_size, device, verbose=False):
        self.dlrm = dlrm
        self.train_ld = train_ld
        self.train_data = train_data
        self.bandmf_solver = bandmf_solver
        self.thresholds = thresholds    # rows accessed > thresholds times are "hot"
        self.ln_emb = ln_emb           # number of embedding tables
        self.m_spa = m_spa             # embedding dimension
        self.epochs = epochs
        self.device = device
        self.noise_std = noise_std     # effective noise std = noise_mult * max_grad_norm
        self.verbose = verbose
        print(f"Noise Std in the noise cache: {noise_std=}")

        # cumul_rows[i] = global row offset where table i starts; shape (ln_emb+1,)
        self.cumul_rows = torch.zeros(ln_emb + 1, dtype=torch.long, device=self.device)
        for i in range(ln_emb):
            self.cumul_rows[i + 1] = dlrm.emb_l[i].weight.size(0) + self.cumul_rows[i]
        print(self.cumul_rows)

        self.steps_per_epoch = len(train_ld)
        self.access_offsets = None     # [step][chunk] → CPU tensor of accessed cold row indices
        self.noise_slices = None       # [chunk][step] → CPU tensor of accumulated noise slice
        self.residual_noise = None     # [chunk]       → CPU tensor for the final step's residual

        self.chunk_size = chunk_size   # max cold rows per chunk (controls GPU memory vs. n_chunks)
        self.n_cold_rows = None        # number of cold rows (access count ≤ threshold)
        self.n_hot_rows = None         # number of hot rows  (access count > threshold)
        self.n_chunks = None           # ceil(n_cold_rows / chunk_size)

        # Count total embedding rows and dense params for noise vector layout
        self.n_emb_rows = self.cumul_rows[-1]           # total rows across all tables
        self.n_emb_params = self.n_emb_rows * m_spa
        self.n_dense_params = 0
        for name, param in dlrm.named_parameters():
            if "emb_l" not in name:
                self.n_dense_params += param.numel()
        print(f"Embedding tables: ({self.n_emb_rows} rows × {m_spa} dim)  "
              f"Dense params: {self.n_dense_params}")

    def build(self):
        """Entry point: classify hot/cold rows, build access index, accumulate BandMF noise."""
        self._build()

    def _build(self):
        """Classify cold rows, build access-pattern index, then accumulate BandMF noise.

        Step 1 — classify: scan all training data to get per-row access counts,
                 split rows into cold (≤ threshold) and hot (> threshold).
        Step 2 — index:    for each training step, record which cold rows are touched
                 (access_offsets), by iterating through train_ld once.
        Step 3 — noise:    for each cold-row chunk, run the full BandMF simulation,
                 accumulating noise and extracting slices at each accessed step.
        """
        def safelen(x):
            return x.size(0) if x.dim() > 0 else 1

        t_start = torch.cuda.Event(enable_timing=True)
        t_start.record()

        # --- Step 1: count per-row accesses across the full dataset ---
        # row_counts[r] = number of times global row r was accessed across all steps
        row_counts = torch.zeros(self.n_emb_rows, dtype=torch.long, device=self.device)

        X_cat_tensor = torch.as_tensor(self.train_data.X_cat, dtype=torch.long).to("cuda:0")
        for i in range(self.ln_emb):
            counts = torch.bincount(X_cat_tensor[:, i])  # access counts for table i
            row_counts[self.cumul_rows[i]:self.cumul_rows[i] + len(counts)] = counts
        del X_cat_tensor

        dist = summarize_frequency_distribution(row_counts)
        print("\n--- Frequency Distribution Summary ---")
        for category, value in dist.items():
            print(f"Items that appeared {category: <12} times: {value: >10,}")

        # cold rows: accessed ≤ threshold → pre-compute noise; hot rows: on-the-fly
        cold_indices = torch.nonzero(row_counts <= self.thresholds, as_tuple=False).squeeze()
        self.hot_indices = torch.nonzero(row_counts > self.thresholds, as_tuple=False).squeeze()
        self.n_cold_rows = len(cold_indices)
        self.n_hot_rows = len(self.hot_indices)
        print(f"Cold rows: {self.n_cold_rows:,}  Hot rows: {self.n_hot_rows:,}")
        assert self.n_emb_rows == self.n_cold_rows + self.n_hot_rows

        t_classified = torch.cuda.Event(enable_timing=True)
        t_classified.record()
        torch.cuda.synchronize()
        print(f"Hot/cold classification time: {t_start.elapsed_time(t_classified):.2f} ms")

        if self.verbose:
            print(f"[verbose] threshold={self.thresholds}  total={self.n_emb_rows:,}")

        # Partition cold rows into chunks for memory-efficient noise accumulation
        self.n_chunks = -(cold_indices.size(0) // -self.chunk_size)  # ceiling division
        self.cold_chunks = torch.split(cold_indices, self.chunk_size)

        if self.verbose:
            print(f"[verbose] n_chunks={self.n_chunks}  "
                  f"chunk sizes: {[len(c) for c in self.cold_chunks]}")

        del cold_indices

        # --- Step 2: build access_offsets for all epochs ---
        # access_offsets[global_step][chunk]: cold rows accessed at that step.
        # global_step 0 is skipped (warm-up); last step has None (→ residual).
        show_memory()
        total_steps = self.epochs * self.steps_per_epoch
        cold_row_counts = []
        self.access_offsets = [None] * total_steps
        global_batch = 0
        for k in range(self.epochs):
            for inputBatch in iter(self.train_ld):
                _1, _2, lS_i, _3, _4, _5 = unpack_batch(inputBatch)
                lS_i = lS_i.to(self.device)
                for tbl in range(self.ln_emb):
                    lS_i[tbl] = lS_i[tbl] + self.cumul_rows[tbl]  # convert to global row IDs
                if global_batch == 0:
                    global_batch += 1
                    continue      # skip very first batch (warm-up offset)
                step = global_batch - 1
                unique_idx = torch.unique(torch.flatten(lS_i))
                self.access_offsets[step] = [None] * self.n_chunks
                step_cold_count = 0
                for ci in range(self.n_chunks):
                    a_cat_b, cnt = torch.cat([self.cold_chunks[ci], unique_idx]).unique(return_counts=True)
                    intersection = a_cat_b[cnt.gt(1)]
                    step_cold_count += intersection.size(0)
                    self.access_offsets[step][ci] = intersection.to('cpu', non_blocking=True)
                cold_row_counts.append(step_cold_count)
                if self.verbose and step % 100 == 0:
                    print(f"[verbose] offset step={step} unique_global={unique_idx.shape[0]} "
                          f"cold_accessed={step_cold_count}")
                global_batch += 1

        t_offsets = torch.cuda.Event(enable_timing=True)
        t_offsets.record()
        torch.cuda.synchronize()
        cold_row_counts = np.array(cold_row_counts)
        print(f"Avg cold rows accessed/step: {np.average(cold_row_counts):.1f} "
              f"± {np.std(cold_row_counts):.1f}  steps: {len(cold_row_counts)}")
        print(f"Offset build time: {t_classified.elapsed_time(t_offsets):.2f} ms")
        show_memory()

        # --- Step 3: simulate full BandMF sequence over all epochs ---
        # For each chunk, accum_noise runs continuously for total_steps.
        # At each step, slice out rows accessed next (access_offsets[step]) and zero them.
        # At the final step, save the remaining accumulation as residual_noise.
        self.noise_slices = [None] * self.n_chunks
        self.residual_noise = [None] * self.n_chunks
        for i in range(self.n_chunks):
            chunk_rows = safelen(self.cold_chunks[i])
            accum_noise = torch.zeros((chunk_rows, self.m_spa), dtype=torch.float, device=self.device)
            self.noise_slices[i] = [None] * total_steps
            for gj in range(total_steps):
                noise = torch.normal(
                    mean=0,
                    std=self.noise_std / self.bandmf_solver.diag(gj),
                    size=(chunk_rows * self.m_spa,),
                    device=self.device,
                )
                if self.bandmf_solver is not None:
                    noise = self.bandmf_solver.step(noise)
                accum_noise += noise.view_as(accum_noise)
                if gj == total_steps - 1:
                    self.residual_noise[i] = accum_noise.to('cpu', non_blocking=True)
                elif self.access_offsets[gj] is not None:
                    self.access_offsets[gj][i] = self.access_offsets[gj][i].to('cuda:0', non_blocking=True)
                    row_idx = torch.searchsorted(self.cold_chunks[i], self.access_offsets[gj][i])
                    self.noise_slices[i][gj] = accum_noise[row_idx].view(-1, self.m_spa)
                    accum_noise[row_idx] = 0
                    self.noise_slices[i][gj] = self.noise_slices[i][gj].to('cpu', non_blocking=True)
                    self.access_offsets[gj][i] = self.access_offsets[gj][i].to('cpu', non_blocking=True)
                if self.verbose and gj % 100 == 0:
                    print(f"[verbose] chunk={i} step={gj} "
                          f"noise shape={tuple(noise.shape)} std={noise.std().item():.4f} "
                          f"accum_norm={accum_noise.norm().item():.4f} "
                          f"diag={self.bandmf_solver.diag(gj):.4f}")
                if self.bandmf_solver is not None:
                    self.bandmf_solver.advance()
            if self.bandmf_solver is not None:
                self.bandmf_solver.reset()

        t_noise = torch.cuda.Event(enable_timing=True)
        t_noise.record()
        torch.cuda.synchronize()
        print(f"access_offsets: {get_memory_usage(self.access_offsets) / 1024**2:.1f} MB")
        print(f"noise_slices:   {get_memory_usage(self.noise_slices) / 1024**2:.1f} MB")
        print(f"residual_noise: {get_memory_usage(self.residual_noise) / 1024**2:.1f} MB")
        print(f"Noise accumulation time: {t_offsets.elapsed_time(t_noise):.2f} ms")
        print(f"Total build time: {t_start.elapsed_time(t_noise):.2f} ms")
        show_memory()
