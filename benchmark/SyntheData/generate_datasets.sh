#!/usr/bin/env bash
# Generate all synthetic Criteo-Kaggle DLRM datasets used in Figures 14–17.
#
# Variants produced:
#   SyntheticKaggleZipf1.npz       alpha=1.0  entry_scale=1.0   (baseline)
#   SyntheticKaggle0_5xZipf1.npz   alpha=1.0  entry_scale=0.5   (half-size tables)
#   SyntheticKaggle2xZipf1.npz     alpha=1.0  entry_scale=2.0   (double-size tables)
#   SyntheticKaggleZipf0_5.npz     alpha=0.5  entry_scale=1.0   (low skew)
#   SyntheticKaggleZipf2.npz       alpha=2.0  entry_scale=1.0   (high skew)
#
# Usage:
#   bash generate_datasets.sh                          # uses defaults below
#   DATA_DIR=/my/data bash generate_datasets.sh        # override output dir
#   KAGGLE_INPUT=/my/data/processed.npz bash ...       # override input file

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNTH_GEN="${SCRIPT_DIR}/synthetic_data_generator.py"
PYTHON="${PYTHON:-python}"

# ── Edit these two paths ──────────────────────────────────────────────────────
# KAGGLE_INPUT : processed Criteo .npz (provides X_int, y, and X_cat shape)
# DATA_DIR     : where to write the synthetic .npz files
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KAGGLE_INPUT="${KAGGLE_INPUT:-${REPO_ROOT}/data/kaggleAdDisplayChallenge_processed.npz}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -f "$KAGGLE_INPUT" ]]; then
    echo "Error: KAGGLE_INPUT not found: $KAGGLE_INPUT"
    echo "Set KAGGLE_INPUT=/path/to/kaggleAdDisplayChallenge_processed.npz"
    exit 1
fi
mkdir -p "$DATA_DIR"

# Each entry: "filename:alpha:entry_scale"
CONFIGS=(
    "SyntheticKaggleZipf1.npz:1.0:1.0"
    "SyntheticKaggle0_5xZipf1.npz:1.0:0.5"
    "SyntheticKaggle2xZipf1.npz:1.0:2.0"
    "SyntheticKaggleZipf0_5.npz:0.5:1.0"
    "SyntheticKaggleZipf2.npz:2.0:1.0"
)

echo "Input : $KAGGLE_INPUT"
echo "Output: $DATA_DIR"
echo ""

for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r filename alpha scale <<< "$entry"
    out="${DATA_DIR}/${filename}"
    if [[ -f "$out" ]]; then
        echo "  (exists)    $filename"
    else
        echo "  Generating  $filename  (alpha=${alpha}  entry_scale=${scale}) ..."
        "$PYTHON" "$SYNTH_GEN" \
            --alpha       "$alpha" \
            --entry-scale "$scale" \
            --input       "$KAGGLE_INPUT" \
            --output      "$out"
        echo "  Done → $out"
    fi
done

echo ""
echo "All datasets ready in $DATA_DIR"
