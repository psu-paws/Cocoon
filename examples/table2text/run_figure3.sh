#!/usr/bin/env bash
# Figure 3: Single iteration execution time vs band size.
#
# For each (model, n_gpu) combo:
#   1. Probe GPU memory at band=1, batch=1 → estimate max viable band size
#   2. Select band points (powers of 2 + near max)
#   3. For each band: binary-search for max viable batch size
#   4. Record step_by_step (ms) from the log
#
# Usage:
#   bash run_figure3.sh                   # all (model, n_gpu) combos
#   bash run_figure3.sh opt350m           # one model, all GPU counts
#   bash run_figure3.sh opt350m 4         # one model, one GPU count

set -uo pipefail

# ── Paths (edit before running) ───────────────────────────────────────────────
PREFIX_PATH="./prefix-tuning"
OUTPUT_PATH="./figure3"
TRAIN_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ZERO1.sh"
DATASET="e2e"
MASTER_PORT=29500

# Physical GPU IDs available (comma-separated, largest-to-smallest index).
ALL_GPU_IDS="4,5,6,7"

# ── Model registry ────────────────────────────────────────────────────────────
# label → HuggingFace model name and total parameter count
declare -A HF_NAME=( [opt350m]="facebook/opt-350m" [opt1.3b]="facebook/opt-1.3b" )

ALL_MODELS=(opt350m opt1.3b)
ALL_GPU_COUNTS=(1 2 4)

# Filter from CLI args
MODELS_TO_RUN=("${ALL_MODELS[@]}")
GPU_COUNTS_TO_RUN=("${ALL_GPU_COUNTS[@]}")
[[ $# -ge 1 ]] && MODELS_TO_RUN=("$1")
[[ $# -ge 2 ]] && GPU_COUNTS_TO_RUN=("$2")

# ── Sweep params ──────────────────────────────────────────────────────────────
PROBE_STEPS=3     # steps for memory probe (band=1, batch=3)
SWEEP_STEPS=10    # steps per timing measurement
MIN_BATCH=3       # minimum physical batch size
MAX_BATCH=512     # upper bound for batch search
TARGET_LOGICAL=1000  # target logical batch size for measurement runs

# ── Helper: first N GPU IDs as comma-separated string ────────────────────────
gpu_ids_for() {
    echo "$ALL_GPU_IDS" | tr ',' '\n' | head -n "$1" | tr '\n' ',' | sed 's/,$//'
}

# ── Helper: run one training job; return 0=success 1=OOM ─────────────────────
# search_mode=1: logical_bs = physical_bs × n_gpu (fast OOM check, no accumulation)
# search_mode=0: logical_bs = nearest multiple of (physical_bs × n_gpu) to TARGET_LOGICAL
run_one() {
    local label=$1 hf_name=$2 n_gpu=$3 physical_bs=$4 band=$5 steps=$6 search_mode=${7:-1}
    local gpu_ids logfile logical_bs unit

    gpu_ids=$(gpu_ids_for "$n_gpu")
    unit=$(( physical_bs * n_gpu ))
    if (( search_mode )); then
        logical_bs=$unit
    else
        logical_bs=$(( ((TARGET_LOGICAL + unit / 2) / unit) * unit ))
        (( logical_bs < unit )) && logical_bs=$unit
    fi
    logfile="${OUTPUT_PATH}/${label}.txt"

    mkdir -p "$OUTPUT_PATH"
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    bash "$TRAIN_SCRIPT" \
        "$PREFIX_PATH" "${OUTPUT_PATH}/ckpt" \
        "$DATASET" "$hf_name" \
        "$physical_bs" "$logical_bs" \
        "no" "$band" \
        "$n_gpu" "$MASTER_PORT" \
        1 1 0 \
        "MixGhostClip" 8 "automatic" "all-layer" \
        "no" "no" "no" "no" \
        "$steps" "True" \
        > "$logfile" 2>&1 || true

    if grep -qi "out of memory" "$logfile"; then
        return 1
    fi
    return 0
}

# ── Helper: extract average step_by_step (ms) across all GPU prints ──────────
extract_iter_ms() {
    local logfile=$1
    grep -oP 'step_by_step\s*:\s*\K[0-9.]+' "$logfile" 2>/dev/null \
        | awk '{s+=$1; n++} END{if(n>0) printf "%.0f", s/n; else print "N/A"}'
}

# ── Helper: binary search for max viable batch at a given band ────────────────
# hint: lower bound from previous band (smaller batch always works if larger did)
find_max_batch() {
    local label=$1 hf_name=$2 n_gpu=$3 band=$4 hint=${5:-$MIN_BATCH}
    local lo=$(( hint - 1 )) hi=$hint

    # Exponential search upward from hint
    while (( hi <= MAX_BATCH )); do
        if run_one "${label}_BAND${band}_BSZ${hi}_search" "$hf_name" "$n_gpu" "$hi" "$band" "$PROBE_STEPS"; then
            lo=$hi
            hi=$(( hi * 2 ))
        else
            break
        fi
    done
    (( hi > MAX_BATCH )) && hi=$(( MAX_BATCH + 1 ))

    # Binary search between lo and hi
    while (( hi - lo > 1 )); do
        local mid=$(( (lo + hi) / 2 ))
        if run_one "${label}_BAND${band}_BSZ${mid}_search" "$hf_name" "$n_gpu" "$mid" "$band" "$PROBE_STEPS"; then
            lo=$mid
        else
            hi=$mid
        fi
    done

    # Ensure minimum batch
    (( lo < MIN_BATCH )) && lo=0
    echo "$lo"
}

# ── Helper: select band points given estimated max_band ──────────────────────
select_bands() {
    local max_band=$1
    local bands=()
    local b=1

    # Powers of 2 up to just before the near-max cluster
    while (( b <= max_band - 5 )); do
        bands+=("$b")
        b=$(( b * 2 ))
    done

    # A few points near max_band to capture the OOM boundary
    for offset in 2 1 0; do
        local pt=$(( max_band - offset ))
        (( pt > 0 )) && bands+=("$pt")
    done

    printf '%s\n' "${bands[@]}" | sort -nu | tr '\n' ' '
}

# ── Main sweep ────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_PATH"
declare -A RESULTS   # key: "label:band" → "max_batch:iter_ms"

for model in "${MODELS_TO_RUN[@]}"; do
    hf_name="${HF_NAME["$model"]}"

    for n_gpu in "${GPU_COUNTS_TO_RUN[@]}"; do
        label="${model}_${n_gpu}GPU"
        gpu_ids=$(gpu_ids_for "$n_gpu")
        first_gpu=$(echo "$gpu_ids" | cut -d',' -f1)

        echo ""
        echo "======================================================"
        echo " $label"
        echo "======================================================"

        # Step 1: probe memory at band=1, batch=3
        echo "[probe] band=1, batch=${MIN_BATCH} for ${PROBE_STEPS} steps ..."
        probe_label="${label}_probe"
        run_one "$probe_label" "$hf_name" "$n_gpu" "$MIN_BATCH" 1 "$PROBE_STEPS" || true

        probe_log="${OUTPUT_PATH}/${probe_label}.txt"
        if grep -qi "out of memory" "$probe_log"; then
            echo "[SKIP] OOM at band=1 batch=${MIN_BATCH} — skipping $label"
            continue
        fi

        # Parse num_params and peak memory from probe log
        num_params=$(grep -oP "'num_params':\s*\K[0-9]+" "$probe_log" | head -1)
        unit_size=$(( num_params / n_gpu ))
        peak_mib=$(grep -oP 'torch\.cuda\.max_memory_allocated:\s*\K[0-9.]+' "$probe_log" \
                   | tail -1 | awk '{printf "%d", $1}')
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
                    -i "$first_gpu" | head -1)
        free_mib=$(( total_mib - peak_mib ))
        free_bytes=$(( free_mib * 1024 * 1024 ))
        max_band=$(( free_bytes / (unit_size * 4) ))
        echo "[probe] num_params=${num_params}  unit_size=${unit_size}  peak=${peak_mib} MiB / total=${total_mib} MiB → free=${free_mib} MiB → est. max_band=${max_band}"

        band_list=$(select_bands "$max_band")
        # Sweep descending: large band (hard) → small band (easy).
        # Max batch can only increase as band decreases, so the hint grows naturally.
        band_list_desc=$(echo "$band_list" | tr ' ' '\n' | sort -rn | tr '\n' ' ')
        echo "[sweep] band points (descending): $band_list_desc"

        # Step 2: for each band, find max batch and measure iter time
        last_max_batch=$MIN_BATCH
        for band in $band_list_desc; do
            echo ""
            echo "  [band=$band] searching for max viable batch (hint=${last_max_batch})..."
            max_batch=$(find_max_batch "$label" "$hf_name" "$n_gpu" "$band" "$last_max_batch")

            if (( max_batch < MIN_BATCH )); then
                echo "  [band=$band] OOM at batch=${MIN_BATCH} — skipping"
                RESULTS["${label}:${band}"]="OOM:OOM"
                continue
            fi

            # Final measurement — binary search if max_batch OOMs (accumulation can expose late OOMs)
            measure_batch=$max_batch
            run_label="${label}_BAND${band}_BSZ${measure_batch}"
            echo "  [band=$band] measuring: batch=${measure_batch}, steps=${SWEEP_STEPS} ..."
            run_one "$run_label" "$hf_name" "$n_gpu" "$measure_batch" "$band" "$SWEEP_STEPS" 0 || true

            if grep -qi "out of memory" "${OUTPUT_PATH}/${run_label}.txt"; then
                echo "  [band=$band] OOM at batch=${measure_batch} during measurement — binary searching"
                bs_lo=$(( MIN_BATCH - 1 ))
                bs_hi=$measure_batch
                while (( bs_hi - bs_lo > 1 )); do
                    bs_mid=$(( (bs_lo + bs_hi) / 2 ))
                    echo "  [band=$band] measuring: batch=${bs_mid}, steps=${SWEEP_STEPS} ..."
                    run_one "${label}_BAND${band}_BSZ${bs_mid}" "$hf_name" "$n_gpu" "$bs_mid" "$band" "$SWEEP_STEPS" 0 || true
                    if grep -qi "out of memory" "${OUTPUT_PATH}/${label}_BAND${band}_BSZ${bs_mid}.txt"; then
                        bs_hi=$bs_mid
                    else
                        bs_lo=$bs_mid
                    fi
                done
                measure_batch=$bs_lo
                run_label="${label}_BAND${band}_BSZ${measure_batch}"
            fi

            if (( measure_batch < MIN_BATCH )); then
                echo "  [band=$band] OOM at all batch sizes during measurement — skipping"
                RESULTS["${label}:${band}"]="OOM:OOM"
                continue
            fi

            iter_ms=$(extract_iter_ms "${OUTPUT_PATH}/${run_label}.txt")
            iter_sec=$(echo "scale=2; ${iter_ms:-0}/1000" | bc 2>/dev/null || echo "N/A")
            echo "  [band=$band] batch=${measure_batch}  iter=${iter_ms} ms (${iter_sec} sec)"
            RESULTS["${label}:${band}"]="${measure_batch}:${iter_ms}"
            last_max_batch=$measure_batch
        done
    done
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " FIGURE 3 RESULTS"
echo "======================================================"
printf "%-22s %-6s %-12s %-12s\n" "Config" "Band" "MaxBatch" "IterTime(sec)"
echo "------------------------------------------------------"
for key in $(echo "${!RESULTS[@]}" | tr ' ' '\n' | sort); do
    IFS=':' read -r cfg band <<< "$key"
    IFS=':' read -r max_batch iter_ms <<< "${RESULTS[$key]}"
    if [[ "$iter_ms" == "OOM" || "$iter_ms" == "N/A" || -z "$iter_ms" ]]; then
        iter_sec="OOM"
    else
        iter_sec=$(echo "scale=2; ${iter_ms}/1000" | bc 2>/dev/null || echo "N/A")
    fi
    printf "%-22s %-6s %-12s %-12s\n" "$cfg" "$band" "$max_batch" "$iter_sec"
done
echo "======================================================"
