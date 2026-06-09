import argparse
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import time

rng = default_rng()

# Per-column maximum entry counts from the Kaggle Criteo dataset (1x baseline)
HI_VALS_1X = {
    0:1460, 1:583, 2:10131227, 3:2202608, 4:305, 5:24, 6:12517, 7:633, 8:3,
    9:93145, 10:5683, 11:8351593, 12:3194, 13:27, 14:14992, 15:5461306, 16:10,
    17:5652, 18:2173, 19:4, 20:7046547, 21:18, 22:15, 23:286181, 24:105, 25:142572
}

def sample_trunc_zipf(hi, n, alpha=1.3):
    ranks = np.arange(1, hi + 1)
    pmf   = 1.0 / np.power(ranks, alpha)
    pmf  /= pmf.sum()
    return rng.choice(hi, size=n, p=pmf, replace=True)

def generate_column_guaranteed(hi, n_rows, alpha):
    if hi > n_rows:
        return sample_trunc_zipf(hi, n_rows, alpha=alpha)
    base  = np.arange(hi, dtype=np.int64)
    extra = n_rows - hi
    col = np.concatenate([base, sample_trunc_zipf(hi, extra, alpha=alpha)]) if extra > 0 else np.copy(base[:n_rows])
    rng.shuffle(col)
    return col

def generate_variable_pooled_column_fast(hi, n_rows, alpha, target_pool_size=2):
    print(f"   hi={hi:,}  alpha={alpha}  pool_size={target_pool_size}")
    total = n_rows * target_pool_size
    feature_pool    = generate_column_guaranteed(hi, total, alpha)
    candidate_matrix = feature_pool.reshape(n_rows, target_pool_size)

    t = time.time()
    sorted_matrix = np.sort(candidate_matrix, axis=1)
    mask = np.concatenate((
        np.full((n_rows, 1), True),
        sorted_matrix[:, 1:] != sorted_matrix[:, :-1]
    ), axis=1)
    unique_rows = [row[m] for row, m in zip(sorted_matrix, mask)]
    print(f"   de-dup: {time.time()-t:.2f}s  avg_unique={np.mean([len(r) for r in unique_rows]):.2f}")
    return np.array(unique_rows, dtype=object)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Criteo-Kaggle DLRM dataset with configurable "
                    "Zipf alpha and embedding-table size scale.")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Zipf distribution exponent (default: 1.0)")
    parser.add_argument("--entry-scale", type=float, default=1.0,
                        help="Scale factor applied to every column's max-entry count "
                             "(0.5 = half-size tables, 2.0 = double-size tables; default: 1.0)")
    parser.add_argument("--output",      type=str,   required=True,
                        help="Path to write the output .npz file")
    parser.add_argument("--input",       type=str,
                        default="../../CriteoDLRM/kaggleAdDisplayChallenge_processed.npz",
                        help="Path to the original processed Criteo npz (provides X_int and y)")
    parser.add_argument("--pool-size",   type=int,   default=2,
                        help="Initial pool size per row before de-duplication (default: 2)")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        print(f"Output already exists: {out_path}  — skipping.")
        return

    # Load continuous features and labels from the real dataset
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Loading {in_path} ...")
    with np.load(in_path) as data:
        X_int  = data['X_int']
        y      = data['y']
        n_rows, n_cat = data['X_cat'].shape

    print(f"  {n_rows:,} rows  {n_cat} categorical columns")
    print(f"  alpha={args.alpha}  entry_scale={args.entry_scale}")

    # Scale hi_vals
    hi_vals = {d: max(1, int(round(v * args.entry_scale)))
               for d, v in HI_VALS_1X.items()}

    arrays_to_save = {'X_int': X_int, 'y': y}
    start = time.time()

    for d in range(n_cat):
        print(f"{'─'*60}\nColumn {d}/{n_cat-1}  hi={hi_vals[d]:,}")
        pooled = generate_variable_pooled_column_fast(
            hi_vals[d], n_rows, args.alpha, args.pool_size)
        lengths = np.array([len(r) for r in pooled], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        indices = np.concatenate(pooled).astype(np.int64)
        arrays_to_save[f'X_cat_{d}_indices'] = indices
        arrays_to_save[f'X_cat_{d}_offsets'] = offsets

    print(f"\nTotal generation: {time.time()-start:.1f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving → {out_path} ...")
    t = time.time()
    np.savez_compressed(str(out_path), **arrays_to_save)
    print(f"Saved in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
