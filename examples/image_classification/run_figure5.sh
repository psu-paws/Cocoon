#!/usr/bin/env bash
# Figure 5: ViT-Large single-iteration time breakdown — DP-SGD vs GPU-GEMV vs CPU-GEMV
#           for 1, 2, and 4 GPUs.  Noise history fits entirely in main memory.
#
# GPU-GEMV breakdown (analytical):
#   Train(GPU)        = DP-SGD baseline (unchanged)
#   GEMV(GPU)         = (TARGET_BAND / max_k) × bench_gemv_gpu(unit_size/n_gpu, max_k)
#   Transfer(MainMem) = (TARGET_BAND - max_k) × (unit_size/n_gpu) × 4B / PCIe_BW
#   where max_k = largest band that fits on GPU alongside the model (found by binary search)
#
# CPU-GEMV:
#   Run with --noise_offload; GEMV is fully hidden behind training (training is tall).
#   Iter time ≈ DP-SGD iter time.
#
# Resume behaviour: if a log file already exists and contains Avg_Iter (or the
# expected bench output), the run is skipped and results are read from the file.
#
# Usage:
#   bash run_figure5.sh              # all GPU counts
#   bash run_figure5.sh 1            # single GPU count

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ══════════════════════════════════════════════════════════════════════════════
# USER-ADJUSTABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PYTHON="${PYTHON:-$(which python)}"
PCIE_BW_GBS=32.0                    # PCIe 4.0 x16 peak bandwidth (GB/s)
TARGET_BAND=16                      # Band used for all three modes
NUM_BATCHES=10                      # Iterations per run (0 = full epoch)
DS_CONFIG="${SCRIPT_DIR}/cifar_config.json"

# Per-GPU-count configs: CUDA devices, numactl CPU list, NUMA memory node
declare -A GPU_DEVICES=( [1]="4"       [2]="4,5"       [4]="4,5,6,7" )
declare -A NUMA_CPUS=(   [1]="28-34,84-90" [2]="28-41,84-97" [4]="28-55,84-111" )
declare -A NUMA_MEM=(    [1]="1"       [2]="1"         [4]="1" )

# ══════════════════════════════════════════════════════════════════════════════

TRAIN_SCRIPT="${SCRIPT_DIR}/CIFAR_TIMM_ZERO1.py"
BENCH_GPU="${SCRIPT_DIR}/../../benchmark/bench_gemv_gpu.py"
OUTPUT="${SCRIPT_DIR}/figure5"
mkdir -p "$OUTPUT"

ALL_NGPUS=(1 2 4)
NGPUS_TO_RUN=("${ALL_NGPUS[@]}")
[[ $# -ge 1 ]] && NGPUS_TO_RUN=("$1")

# Count CPU cores from numactl spec (e.g. "0-6,56-62" → 14)
_count_cpus() {
    local n=0; IFS=',' read -ra parts <<< "$1"
    for p in "${parts[@]}"; do
        if [[ "$p" == *-* ]]; then n=$(( n + ${p#*-} - ${p%-*} + 1 ))
        else n=$(( n + 1 )); fi
    done; echo $n
}

# Run deepspeed training, tee to logfile; skip if logfile already has Avg_Iter
run_train() {
    local logfile=$1; shift
    local ngpu=$1; shift
    mkdir -p "$(dirname "$logfile")"
    if [[ -f "$logfile" ]] && grep -q 'Avg_Iter' "$logfile" 2>/dev/null; then
        echo "  (skipping — $logfile already has Avg_Iter)"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$ngpu]}" \
    numactl -C "${NUMA_CPUS[$ngpu]}" -m "${NUMA_MEM[$ngpu]}" \
    deepspeed "$TRAIN_SCRIPT" \
        --deepspeed_config "${DS_CONFIG}" \
        --model vit_large_patch16_224 \
        --num_batches "$NUM_BATCHES" \
        "$@" 2>&1 | tee "$logfile"
}

# Parse Avg_Iter from log (ms); returns empty string if not found
extract_avg_iter() {
    grep -oP 'Avg_Iter \K[0-9.]+' "$1" 2>/dev/null | tail -1
}

# Compute transfer latency in ms
xfer_ms() {
    echo "scale=2; $1 / ($PCIE_BW_GBS * 1000000000) * 1000" | bc
}

declare -A RESULTS      # ngpu:mode → train_ms:xfer_ms:gemv_ms
declare -A MAX_BAND     # ngpu → max band that fits in GPU
UNIT_SIZE=304326632     # ViT-Large trainable params

# ── Phase 1: DP-SGD baseline (band=1) for each GPU count ─────────────────────
echo ""
echo "======================================================"
echo " Phase 1: DP-SGD (band=1)"
echo "======================================================"
for ngpu in "${NGPUS_TO_RUN[@]}"; do
    echo ""
    echo "  [DP-SGD ngpu=${ngpu}]"
    logfile="${OUTPUT}/dpsgd_gpu${ngpu}.txt"
    run_train "$logfile" "$ngpu" --min_separation 1 || true
    train_ms=$(extract_avg_iter "$logfile")
    echo "  train=${train_ms:-N/A}ms"
    RESULTS["${ngpu}:dpsgd"]="${train_ms:-N/A}:0:0"
done

# ── Phase 2: CPU-GEMV (noise_offload, band=TARGET_BAND) ──────────────────────
echo ""
echo "======================================================"
echo " Phase 2: CPU-GEMV (--noise_offload, band=${TARGET_BAND})"
echo "======================================================"
for ngpu in "${NGPUS_TO_RUN[@]}"; do
    echo ""
    echo "  [CPU-GEMV ngpu=${ngpu} band=${TARGET_BAND}]"
    logfile="${OUTPUT}/cpugemv_gpu${ngpu}_b${TARGET_BAND}.txt"
    num_threads=$(_count_cpus "${NUMA_CPUS[$ngpu]}")
    run_train "$logfile" "$ngpu" \
        --min_separation "$TARGET_BAND" \
        --noise_offload \
        --noise-num-threads "$num_threads" || true
    train_ms=$(extract_avg_iter "$logfile")
    echo "  train=${train_ms:-N/A}ms  (GEMV hidden behind GPU compute)"
    RESULTS["${ngpu}:cpugemv"]="${train_ms:-N/A}:0:0"
done

# ── Phase 3: Find max GPU band (binary search) + GPU-GEMV analytical ─────────
echo ""
echo "======================================================"
echo " Phase 3: GPU-GEMV — binary search for max in-GPU band"
echo "======================================================"
for ngpu in "${NGPUS_TO_RUN[@]}"; do
    unit_per_gpu=$(( UNIT_SIZE / ngpu ))
    dpsgd_train=$(echo "${RESULTS[${ngpu}:dpsgd]:-N/A:0:0}" | cut -d: -f1)

    # Binary search: find largest band that runs without OOM.
    # Probe bs_hi (TARGET_BAND) first — if it fits, we're done immediately.
    # On OOM, binary search between 1 and TARGET_BAND-1.
    # Existing probe logs are reused without re-running.
    bs_lo=1
    bs_hi=$TARGET_BAND
    max_k=1

    _probe() {
        local band=$1
        local lf="${OUTPUT}/gpugemv_probe_gpu${ngpu}_b${band}.txt"
        echo ""
        echo "  [GPU-GEMV probe ngpu=${ngpu} band=${band}]"
        if [[ -f "$lf" ]]; then
            echo "  (reading from existing $lf)"
        else
            run_train "$lf" "$ngpu" --min_separation "$band" || true
        fi
        grep -qi "out of memory" "$lf" 2>/dev/null && return 1 || return 0
    }

    if _probe "$bs_hi"; then
        echo "  → OK"
        max_k=$bs_hi
    else
        echo "  → OOM"
        bs_hi=$(( bs_hi - 1 ))
        while (( bs_hi - bs_lo > 0 )); do
            bs_mid=$(( (bs_lo + bs_hi + 1) / 2 ))
            if _probe "$bs_mid"; then
                echo "  → OK"
                max_k=$bs_mid
                bs_lo=$bs_mid
            else
                echo "  → OOM"
                bs_hi=$(( bs_mid - 1 ))
            fi
        done
        # Final check on the remaining candidate
        if (( bs_lo == bs_hi )) && _probe "$bs_lo"; then
            echo "  → OK"
            max_k=$bs_lo
        fi
    fi
    MAX_BAND["$ngpu"]=$max_k
    echo ""
    echo "  [GPU-GEMV ngpu=${ngpu}] max_k=${max_k} / target=${TARGET_BAND}"

    # bench_gemv_gpu for the max fitting band; skip if log already has avg result
    bench_logfile="${OUTPUT}/bench_gpu_gpu${ngpu}_k${max_k}.txt"
    if [[ -f "$bench_logfile" ]] && grep -q 'avg' "$bench_logfile" 2>/dev/null; then
        echo "  (reading bench from existing $bench_logfile)"
    else
        echo "  Running bench_gemv_gpu(unit_size=${unit_per_gpu}, band=${max_k})..."
        CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$ngpu]%,*}" \
        "$PYTHON" "$BENCH_GPU" "$unit_per_gpu" "$max_k" 2>&1 | tee "$bench_logfile" || true
    fi
    gemv_k_ms=$(grep -oP 'avg \K[0-9.]+(?= ms)' "$bench_logfile" | tail -1)

    # Scale GEMV and compute transfer for overflow bands
    if [[ "$gemv_k_ms" =~ ^[0-9.]+$ && "$max_k" -gt 0 ]]; then
        gemv_ms=$(echo "scale=2; ($TARGET_BAND / $max_k) * $gemv_k_ms" | bc)
    else
        gemv_ms="N/A"
    fi
    overflow=$(( TARGET_BAND - max_k ))
    xfer_bytes=$(echo "scale=0; $overflow * $unit_per_gpu * 4 / 1" | bc)
    xfer_ms_val=$(xfer_ms "$xfer_bytes")

    echo "  GEMV=${gemv_ms}ms  Transfer=${xfer_ms_val}ms  (max_k=${max_k} overflow=${overflow})"
    RESULTS["${ngpu}:gpugemv"]="${dpsgd_train}:${xfer_ms_val}:${gemv_ms}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================="
echo " FIGURE 5 RESULTS — ViT-Large Single-Iteration Time Breakdown (ms)"
echo " (band=${TARGET_BAND}  GPU-GEMV=analytical  CPU-GEMV=measured)"
echo "========================================================================="
printf "%-6s %-10s %12s %12s %12s %12s\n" \
    "nGPU" "Mode" "Train(ms)" "Transfer(ms)" "GEMV(ms)" "Total(ms)"
echo "-------------------------------------------------------------------------"
for ngpu in "${ALL_NGPUS[@]}"; do
    _any=0
    for mode in dpsgd cpugemv gpugemv; do
        [[ "${RESULTS[${ngpu}:${mode}]+x}" ]] && _any=1
    done
    (( _any == 0 )) && continue
    for mode in dpsgd cpugemv gpugemv; do
        IFS=':' read -r tr tx gv <<< "${RESULTS[${ngpu}:${mode}]:-N/A:0:0}"
        if [[ "$tr" =~ ^[0-9.]+$ && "$tx" =~ ^[0-9.]+$ && "$gv" =~ ^[0-9.]+$ ]]; then
            total=$(echo "scale=2; $tr + $tx + $gv" | bc)
        else
            total="N/A"
        fi
        printf "%-6s %-10s %12s %12s %12s %12s\n" \
            "$ngpu" "$mode" "${tr:-N/A}" "${tx:-0}" "${gv:-0}" "$total"
    done
    echo "-------------------------------------------------------------------------"
done
echo "========================================================================="
