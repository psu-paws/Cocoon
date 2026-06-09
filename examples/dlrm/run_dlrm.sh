#!/usr/bin/env bash
# Figures 14–15: DLRM training time breakdown and speedup analysis.
#
# Figure 14 : Normalized training time vs band (d_emb=16, 1x Zipf1, B=65536)
# Figure 15a: Speedup vs band, varying d_emb (8, 16, 32)
# Figure 15b: Speedup vs band, varying embedding entries (0.5x, 1x, 2x Zipf1)
# Figure 15c: Speedup vs band, varying batch size (32K, 64K, 128K)
# Figure 15d: Speedup vs band, varying skewness (Zipf α=0.5, 1.0, 2.0)
#
# GPU-GEMV is analytical:
#   max_band = floor(GPU_free_MB × 1e6 / unit_bytes)
#   t_precompute = t_gemv_bench × (B-1) / (max_band-1)  [scaled]
#   t_transfer   = max(0, B - max_band) × unit_bytes / PCIe_BW
#
# Usage:
#   bash run_dlrm.sh          # all figures
#   bash run_dlrm.sh 14       # only Fig 14
#   bash run_dlrm.sh 15a      # only Fig 15a  (etc.)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLRM_SCRIPT="${SCRIPT_DIR}/dlrm_s_pytorch.py"
BENCH_GPU="${SCRIPT_DIR}/../../benchmark/bench_gemv_gpu.py"
BENCH_CPU="${SCRIPT_DIR}/../../benchmark/bench_gemv_cpu.py"

# ══════════════════════════════════════════════════════════════════════════════
# USER-ADJUSTABLE
# ══════════════════════════════════════════════════════════════════════════════
GPU="${GPU:-3}"
NUMA_CPUS="${NUMA_CPUS:-21-27,77-83}"
NUMA_MEM="${NUMA_MEM:-0}"
PYTHON="${PYTHON:-python}"

GPU_TOTAL_MB=24564        # RTX A5000 VRAM
GPU_OVERHEAD_MB=2048      # activations + gradients + optimizer states
BW_CPUGPU_GBS=32.0        # PCIe 4.0 x16 peak bandwidth (GB/s)

# ── Dataset root (edit this one path to point at your Criteo data) ───────────
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export CRITEO_DATA_DIR="$DATA_DIR"

RAW_DATA="${DATA_DIR}/train.txt"
KAGGLE_PROC="${DATA_DIR}/kaggleAdDisplayChallenge_processed_fast.npz"
ZIPF1_1X="${DATA_DIR}/SyntheticKaggleZipf1.npz"
ZIPF1_05X="${DATA_DIR}/SyntheticKaggle0_5xZipf1.npz"
ZIPF1_2X="${DATA_DIR}/SyntheticKaggle2xZipf1.npz"
ZIPF05="${DATA_DIR}/SyntheticKaggleZipf0_5.npz"
ZIPF2="${DATA_DIR}/SyntheticKaggleZipf2.npz"

SYNTH_GEN="${SCRIPT_DIR}/../../benchmark/SyntheData/synthetic_data_generator.py"
KAGGLE_INPUT="${DATA_DIR}/kaggleAdDisplayChallenge_processed.npz"

ALL_BANDS=(2 4 8 16 32 64)
OUTPUT="${SCRIPT_DIR}/figure14-15"
mkdir -p "$OUTPUT"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Generate synthetic datasets (skip if files already exist)
# ══════════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════"
echo " Phase 0 — Synthetic dataset generation"
echo "══════════════════════════════════════════════════════════"

# Each entry: "output_path:alpha:entry_scale"
declare -A SYNTH_CONFIGS=(
    ["$ZIPF1_1X"]="1.0:1.0"
    ["$ZIPF1_05X"]="1.0:0.5"
    ["$ZIPF1_2X"]="1.0:2.0"
    ["$ZIPF05"]="0.5:1.0"
    ["$ZIPF2"]="2.0:1.0"
)

for out_path in "$ZIPF1_1X" "$ZIPF1_05X" "$ZIPF1_2X" "$ZIPF05" "$ZIPF2"; do
    IFS=':' read -r alpha scale <<< "${SYNTH_CONFIGS[$out_path]}"
    if [[ -f "$out_path" ]]; then
        echo "  (exists) $out_path"
    else
        echo "  Generating $(basename "$out_path")  alpha=${alpha}  entry_scale=${scale} ..."
        "$PYTHON" "$SYNTH_GEN" \
            --alpha       "$alpha" \
            --entry-scale "$scale" \
            --input       "$KAGGLE_INPUT" \
            --output      "$out_path"
    fi
done
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# WHICH FIGURES TO RUN
# ══════════════════════════════════════════════════════════════════════════════
RUN_FIGS=("14" "15a" "15b" "15c" "15d")
[[ $# -ge 1 ]] && RUN_FIGS=("$1")
should_run() { local f; for f in "${RUN_FIGS[@]}"; do [[ "$f" == "$1" ]] && return 0; done; return 1; }

# ══════════════════════════════════════════════════════════════════════════════
# FIXED TRAINING ARGS
# ══════════════════════════════════════════════════════════════════════════════
FIXED_ARGS=(
    --arch-mlp-top="512-256-1"
    --data-generation=dataset
    --data-set=kaggle
    --raw-data-file="$RAW_DATA"
    --loss-function=bce
    --round-targets=True
    --learning-rate=1.0
    --print-freq=10000
    --print-time
    --use-gpu
    --nepochs=1
    --numpy-rand-seed=42
    --target-epsilon=3.0
    --target-delta=2.55e-8
    --max_grad_norm=30
    --thresholds=5
    --num-workers=6
    --enable-profiling
    --speed-mode
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Chunk size (rows) for preprocessing given total embedding rows and band.
# Values match test.sh and test2.sh for 1x scale; scale proportionally for others.
chunk_for() {
    local rows=$1 band=$2
    case "$band" in
        2)  echo "$rows" ;;
        4)  echo "$rows" ;;
        8)  echo $(( rows / 2 )) ;;
        16) echo $(( rows / 2 )) ;;
        32) echo $(( rows / 4 )) ;;
        64) echo $(( rows / 7 )) ;;
        *)  echo $(( rows / band )) ;;
    esac
}

# Run one training job; skip if log already has Avg_Iter.
# run <logfile> <proc_data> <d_emb> <batch> <min_sep> <chunk|0> [extra args...]
run() {
    local logfile=$1 proc_data=$2 d_emb=$3 batch=$4 min_sep=$5 chunk=$6
    shift 6
    mkdir -p "$(dirname "$logfile")"
    if [[ -f "$logfile" ]] && grep -q 'Avg_Iter' "$logfile" 2>/dev/null; then
        echo "  (reuse $logfile)"
        return 0
    fi
    local pp_args=()
    [[ "$chunk" -gt 0 ]] && pp_args=(--preprocessing "--chunk-size=$chunk")
    CUDA_VISIBLE_DEVICES="$GPU" \
    numactl -C "$NUMA_CPUS" -m "$NUMA_MEM" \
    "$PYTHON" "$DLRM_SCRIPT" \
        "${FIXED_ARGS[@]}" \
        "--processed-data-file=$proc_data" \
        "--arch-sparse-feature-size=$d_emb" \
        "--arch-mlp-bot=13-512-256-64-${d_emb}" \
        "--batch-size=$batch" \
        "--mini-batch-size=$batch" \
        "--test-mini-batch-size=$batch" \
        "--min-separation=$min_sep" \
        "${pp_args[@]}" \
        "$@" \
        2>&1 | tee "$logfile"
}

# Extract Avg_Iter (ms) from log
avg_iter()      { grep -oP 'Avg_Iter \K[0-9.]+' "$1" 2>/dev/null | tail -1; }
avg_noise_gen() { grep -oP 'Avg_Noise_Gen \K[0-9.]+' "$1" 2>/dev/null | tail -1; }
# Compute steps per epoch: ceil(shuffled_indices / logical_batch)
n_steps_from_log() {
    local shuf batch
    shuf=$(grep -oP 'Shuffled indices:\s*\K[0-9]+' "$1" 2>/dev/null | tail -1)
    batch=$(grep -oP 'logical_batch=\K[0-9]+' "$1" 2>/dev/null | tail -1)
    [[ "$shuf" =~ ^[0-9]+$ && "$batch" =~ ^[0-9]+$ ]] && \
        python3 -c "import math; print(math.ceil($shuf / $batch))"
}
avg_noise_add() { grep -oP 'Avg_Noise_Add \K[0-9.]+' "$1" 2>/dev/null | tail -1; }

# Extract avg_epoch (ms) from log
avg_epoch() { grep -oP 'avg_epoch\s+\K[0-9.]+' "$1" 2>/dev/null | tail -1; }

# Extract "Total build time" (ms) from preprocessing log (0 if absent)
total_build_ms() { grep -oP 'Total build time:\s*\K[0-9.]+' "$1" 2>/dev/null | tail -1; }

# Extract cold_emb bytes from [noise layout] line (strip commas from number)
cold_emb_bytes() { grep -oP 'cold_emb=\K[0-9,]+' "$1" 2>/dev/null | tail -1 | tr -d ','; }

# Extract total noise params from [noise layout] total= field
total_params() { grep -oP '\[noise layout\] total=\K[0-9,]+' "$1" 2>/dev/null | tail -1 | tr -d ','; }

# Count CPU cores from a NUMA spec like "28-34,84-90"
count_cpus() {
    local n=0; IFS=',' read -ra parts <<< "$1"
    for p in "${parts[@]}"; do
        if [[ "$p" == *-* ]]; then n=$(( n + ${p#*-} - ${p%-*} + 1 ))
        else n=$(( n + 1 )); fi
    done; echo $n
}

# run_probe: training run for OOM detection; skips if log already has pass or OOM result.
# Unlike run(), does NOT require preprocessing and skips on either Avg_Iter or OOM in log.
# Args: logfile proc_data d_emb batch min_sep
run_probe() {
    local logfile=$1 proc_data=$2 d_emb=$3 batch=$4 min_sep=$5
    mkdir -p "$(dirname "$logfile")"
    if [[ -f "$logfile" ]] && \
       { grep -q 'Avg_Iter' "$logfile" || grep -qi 'out of memory\|RuntimeError\|CUDA error' "$logfile"; } 2>/dev/null; then
        echo "    (reuse probe b=${min_sep})"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$GPU" \
    numactl -C "$NUMA_CPUS" -m "$NUMA_MEM" \
    "$PYTHON" "$DLRM_SCRIPT" \
        "${FIXED_ARGS[@]}" \
        "--processed-data-file=$proc_data" \
        "--arch-sparse-feature-size=$d_emb" \
        "--arch-mlp-bot=13-512-256-64-${d_emb}" \
        "--batch-size=$batch" \
        "--mini-batch-size=$batch" \
        "--test-mini-batch-size=$batch" \
        "--min-separation=$min_sep" \
        2>&1 | tee "$logfile" || true
}
probe_passed() { grep -q 'Avg_Iter' "$1" 2>/dev/null; }

# find_max_passed_band: binary search for largest band that fits in GPU VRAM without OOM.
# Sets globals: _max_passed_band, _closest_band (largest in ALL_BANDS ≤ max), _closest_log.
# Args: logdir proc_data d_emb batch
find_max_passed_band() {
    local logdir=$1 proc_data=$2 d_emb=$3 batch=$4
    local lo=1 hi=0

    # Phase 1: traverse 1,2,4,8,... to bracket [lo, hi]
    for B in 1 2 4 8 16 32 64; do
        local plog="${logdir}/probe_b${B}.txt"
        echo "    probe band=${B}..."
        run_probe "$plog" "$proc_data" "$d_emb" "$batch" "$B"
        if probe_passed "$plog"; then
            lo=$B
        else
            hi=$B
            break
        fi
    done

    if [[ "$hi" -eq 0 ]]; then
        _max_passed_band=$lo
        echo "  All probed bands passed; max_passed_band=${_max_passed_band}"
    else
        # Phase 2: binary search between lo and hi
        echo "  Bracket [${lo},${hi}]; binary searching..."
        while [[ $(( hi - lo )) -gt 1 ]]; do
            local mid=$(( (lo + hi) / 2 ))
            local plog="${logdir}/probe_b${mid}.txt"
            echo "    probe band=${mid}..."
            run_probe "$plog" "$proc_data" "$d_emb" "$batch" "$mid"
            if probe_passed "$plog"; then lo=$mid; else hi=$mid; fi
        done
        _max_passed_band=$lo
        echo "  max_passed_band=${_max_passed_band} (OOM at band=${hi})"
    fi

    # _closest_band: largest band in ALL_BANDS that is ≤ _max_passed_band
    _closest_band=1
    _closest_log="${logdir}/probe_b1.txt"
    for B in "${ALL_BANDS[@]}"; do
        if [[ "$B" -le "$_max_passed_band" ]]; then
            _closest_band=$B
            _closest_log="${logdir}/probe_b${B}.txt"
        fi
    done
    echo "  closest_band=${_closest_band}"
}

# Bench GPU GEMV and compute analytical GPU-GEMV time components.
# Sets globals: _t_precompute _t_transfer _max_band
# Args: unit_params band bench_log
bench_gpu_gemv() {
    local unit_params=$1 band=$2 bench_log=$3
    local unit_bytes=$(( unit_params * 4 ))
    local gpu_free_bytes=$(python3 -c "print(int(($GPU_TOTAL_MB - $GPU_OVERHEAD_MB) * 1e6))")
    _max_band=$(python3 -c "print(max(1, int($gpu_free_bytes / $unit_bytes)))")
    local fitting=$(( _max_band - 1 ))
    local band_minus1=$(( band - 1 ))

    # Cap bench size to avoid OOM on large embedding tables.
    # GEMV time scales linearly with params, so we extrapolate from a smaller bench.
    local BENCH_MAX_PARAMS=8000000
    local bench_params; bench_params=$(python3 -c "print(min($unit_params, $BENCH_MAX_PARAMS))")
    local bench_bytes=$(( bench_params * 4 ))
    local bench_max_band=$(python3 -c "print(max(2, int($gpu_free_bytes / $bench_bytes)))")

    mkdir -p "$(dirname "$bench_log")"
    if [[ ! -f "$bench_log" ]] || ! grep -q 'avg' "$bench_log" 2>/dev/null; then
        echo "  bench_gemv_gpu(params=${bench_params}, max_band=${bench_max_band})..."
        CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" "$BENCH_GPU" "$bench_params" "$bench_max_band" 2>&1 | tee "$bench_log" || true
    fi
    local t_gemv
    t_gemv=$(grep -oP 'avg \K[0-9.]+(?= ms)' "$bench_log" 2>/dev/null | tail -1)

    if [[ "$band_minus1" -eq 0 ]]; then
        _t_precompute="0.00"; _t_transfer="0.00"
    elif [[ "$fitting" -le 0 ]]; then
        # Nothing fits in GPU beyond model — no GPU GEMV, all overflow to transfer
        _t_precompute="0.00"
    elif [[ "$t_gemv" =~ ^[0-9.]+$ ]]; then
        # Scale bench result to actual (unit_params, band): linear in both params and band count
        _t_precompute=$(python3 -c "print(f'{$t_gemv * ($unit_params / $bench_params) * $band_minus1 / ($bench_max_band - 1):.2f}')")
    else
        _t_precompute="N/A"
    fi

    local overflow=$(( band_minus1 > fitting ? band_minus1 - fitting : 0 ))
    _t_transfer=$(python3 -c "print(f'{$overflow * $unit_bytes / ($BW_CPUGPU_GBS * 1e9) * 1000:.2f}')")
}

# Bench CPU GEMV. Sets global: _t_cpu_gemv_ms (GEMV-only ms/step, excluding H2D).
# CPU-GEMV is pipelined with GPU training; overhead = max(0, t_cpu_gemv - t_iter).
# Args: unit_params band bench_log
bench_cpu_gemv() {
    local unit_params=$1 band=$2 bench_log=$3
    local num_threads
    num_threads=$(count_cpus "$NUMA_CPUS")
    mkdir -p "$(dirname "$bench_log")"
    if [[ ! -f "$bench_log" ]] || ! grep -q 'GEMV' "$bench_log" 2>/dev/null; then
        echo "  bench_gemv_cpu(params=${unit_params}, threads=${num_threads}, band=${band})..."
        numactl -C "$NUMA_CPUS" -m "$NUMA_MEM" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" "$BENCH_CPU" "$unit_params" "$num_threads" "$band" 2>&1 | tee "$bench_log" || true
    fi
    _t_cpu_gemv_ms=$(grep -oP 'GEMV \K[0-9.]+(?= ms)' "$bench_log" 2>/dev/null | tail -1)
}

# Compute speedup of Cocoon over the better baseline (GPU-GEMV vs CPU-GEMV).
# Sets globals: _speedup _t_gpugemv_total _t_cpugemv_total
#   GPU-GEMV total = t_dpsgd + n_iters × (t_precompute + t_transfer)   [analytical]
#   CPU-GEMV total = t_dpsgd + n_iters × max(0, t_cpu_gemv - t_iter)   [pipelined]
#   speedup        = min(t_gpugemv_total, t_cpugemv_total) / t_cocoon_total
# Args: cocoon_total t_dpsgd n_iters t_iter_ms
compute_speedup() {
    local cocoon_total=$1 t_dpsgd=$2 n_iters=$3 t_iter_ms=$4
    _t_gpugemv_total="N/A"; _t_cpugemv_total="N/A"; _speedup="N/A"
    [[ "$t_dpsgd" =~ ^[0-9.]+$ && "$n_iters" =~ ^[0-9]+$ && "$_t_precompute" =~ ^[0-9.]+$ ]] && \
        _t_gpugemv_total=$(python3 -c "print(f'{$t_dpsgd + $n_iters * ($_t_precompute + $_t_transfer):.1f}')")
    [[ "$t_dpsgd" =~ ^[0-9.]+$ && "$n_iters" =~ ^[0-9]+$ && \
       "$t_iter_ms" =~ ^[0-9.]+$ && "$_t_cpu_gemv_ms" =~ ^[0-9.]+$ ]] && \
        _t_cpugemv_total=$(python3 -c "print(f'{$t_dpsgd + $n_iters * max(0.0, $_t_cpu_gemv_ms - $t_iter_ms):.1f}')")
    if [[ "$cocoon_total" =~ ^[0-9.]+$ ]]; then
        if [[ "$_t_gpugemv_total" =~ ^[0-9.]+$ && "$_t_cpugemv_total" =~ ^[0-9.]+$ ]]; then
            _speedup=$(python3 -c "print(f'{min($_t_gpugemv_total, $_t_cpugemv_total) / $cocoon_total:.3f}')")
        elif [[ "$_t_gpugemv_total" =~ ^[0-9.]+$ ]]; then
            _speedup=$(python3 -c "print(f'{$_t_gpugemv_total / $cocoon_total:.3f}')")
        fi
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Normalized training time vs band (d_emb=16, kaggle-processed, B=65536)
#
# Normalization base: DP-SGD avg_epoch (band=1).
#
# GPU-GEMV (all noise on GPU, PCIe overflow for large bands):
#   fitting = max_passed_band (empirical, from OOM binary search)
#   GPU_free = fitting × unit_params × 4 bytes
#   params_gpu(B) = min(unit_params, GPU_free / ((B-1)×4))
#   t_gpu_gemv = t_bench_gpu × (unit_params / params_gpu)   [= t_bench × (B-1)/fitting]
#   t_h2d = max(0, B-1-fitting) × unit_params × 4B / BW    [PCIe overflow]
#   per_iter = t_compute + t_gpu_gemv + t_h2d + t_noise_add
#
# CPU-GEMV (params split: params_gpu on GPU, params_cpu on CPU):
#   t_gpu_gemv_cpu = t_bench_gpu                            [constant: GPU always fills memory]
#   t_cpu_gemv = bench_gemv_cpu(params_cpu, B)              [includes GEMV + H2D per bench design]
#   effective = max(t_compute + t_bench_gpu, t_cpu_gemv) + t_noise_add
#
# t_compute and t_noise_add: from closest passing band log (closest in ALL_BANDS ≤ fitting).
#
# Cocoon: avg_epoch + Total build time (preprocessing cost, once per epoch).
# ══════════════════════════════════════════════════════════════════════════════
if should_run 14 ; then
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 14 — kaggle-processed, d_emb=16, B=65536"
    echo "══════════════════════════════════════════════════════════"
    F14="${OUTPUT}/fig14"
    mkdir -p "$F14"

    # ─── OOM binary search (probe_b1 doubles as the DP-SGD baseline) ────────
    F14_PROBE="${F14}/probe"
    mkdir -p "$F14_PROBE"
    echo "  [OOM search — empirical max GPU band; probe_b1 = DP-SGD baseline]"
    find_max_passed_band "$F14_PROBE" "$KAGGLE_PROC" 16 65536
    fitting14=$_max_passed_band            # GPU holds this many unit_params history vectors

    # probe_b1.txt is the band=1 (DP-SGD) full training run produced above
    F14_DPSGD="${F14_PROBE}/probe_b1.txt"

    t14_iter=$(avg_iter "$F14_DPSGD")
    t14_epoch=$(avg_epoch "$F14_DPSGD")
    echo "  avg_iter=${t14_iter:-N/A}ms  avg_epoch=${t14_epoch:-N/A}ms"

    n14_iters=$(n_steps_from_log "$F14_DPSGD")
    n14_iters="${n14_iters:-N/A}"

    emb_rows_14=$(grep -oP '\[noise layout\].*emb=\d+ \(\K[0-9]+(?= rows)' "$F14_DPSGD" 2>/dev/null | tail -1)
    [[ -z "$emb_rows_14" ]] && emb_rows_14=33762577
    unit_params_14=$(( emb_rows_14 * 16 ))
    echo "  emb_rows=${emb_rows_14}  unit_params=${unit_params_14}  n_iters=${n14_iters}"
    gpu14_free_bytes=$(python3 -c "print($fitting14 * $unit_params_14 * 4)")
    echo "  fitting=${fitting14}  GPU_free=$(python3 -c "print(f'{$gpu14_free_bytes/1e9:.2f}')")GB"

    # t_compute and t_noise_add from the closest passing band probe log.
    # Probe runs have no preprocessing, so Avg_Noise_Gen = iid gen + BandMF GEMV only
    # (no cold H2D transfer). Using closest band gives t_compute at realistic band size.
    t14_compute="N/A"; t14_noise_add_val="N/A"
    _ngen=$(avg_noise_gen "$_closest_log" 2>/dev/null)
    _nadd=$(avg_noise_add "$_closest_log" 2>/dev/null)
    _iter_c=$(avg_iter    "$_closest_log" 2>/dev/null)
    [[ "$_iter_c" =~ ^[0-9.]+$ && "$_ngen" =~ ^[0-9.]+$ && "$_nadd" =~ ^[0-9.]+$ ]] && \
        t14_compute=$(python3 -c "print(f'{$_iter_c - $_ngen - $_nadd:.3f}')")
    [[ "$_nadd" =~ ^[0-9.]+$ ]] && t14_noise_add_val="$_nadd"
    echo "  t_compute=${t14_compute:-N/A}ms  t_noise_add=${t14_noise_add_val:-N/A}ms  (band=${_closest_band})"

    # ─── Cocoon (preprocessing) for each band ────────────────────────────────
    declare -A t14_cocoon t14_build
    for B in "${ALL_BANDS[@]}"; do
        chunk=$(chunk_for "$emb_rows_14" "$B")
        CLOG="${F14}/cocoon_b${B}.txt"
        echo "  [Cocoon band=${B}]"
        run "$CLOG" "$KAGGLE_PROC" 16 65536 "$B" "$chunk"
        t14_cocoon[$B]=$(avg_epoch "$CLOG")
        bms=$(total_build_ms "$CLOG"); t14_build[$B]="${bms:-0}"
        echo "    avg_epoch=${t14_cocoon[$B]:-N/A}ms  build=${t14_build[$B]}ms"
    done

    # ─── GPU-GEMV bench at (unit_params, fitting) — one bench, scaled per band ─
    F14_BENCH_GPU="${F14}/bench_gpu_f${fitting14}.txt"
    if [[ ! -f "$F14_BENCH_GPU" ]] || ! grep -q 'avg' "$F14_BENCH_GPU" 2>/dev/null; then
        echo "  [GPU-GEMV bench unit_params=${unit_params_14} fitting=${fitting14}]"
        CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" "$BENCH_GPU" "$unit_params_14" "$fitting14" 2>&1 | tee "$F14_BENCH_GPU" || true
    fi
    t14_bench_gpu=$(grep -oP 'avg \K[0-9.]+(?= ms)' "$F14_BENCH_GPU" 2>/dev/null | tail -1)
    echo "  t_bench_gpu=${t14_bench_gpu:-N/A}ms"

    # ─── CPU-GEMV bench: one per band, at params_cpu(B) for that band ────────
    # Noise buffer size = B vectors (not B-1); so:
    #   params_gpu(B) = GPU_free / (B×4)  — GPU holds all B bands for params_gpu params
    #   t_gpu_gemv_cpu = t_bench_gpu (constant: GPU always fills its memory exactly)
    # bench_gemv_cpu includes GEMV + H2D.
    cpu14_threads=$(count_cpus "$NUMA_CPUS")
    echo "  [CPU-GEMV bench threads=${cpu14_threads}]"
    declare -A t14_params_gpu t14_params_cpu t14_cpu_gemv
    for B in "${ALL_BANDS[@]}"; do
        bm1=$(( B - 1 ))
        if [[ "$bm1" -le 0 ]]; then
            t14_params_gpu[$B]=$unit_params_14; t14_params_cpu[$B]=0; t14_cpu_gemv[$B]="0.000"
            continue
        fi
        pg=$(python3 -c "print(min($unit_params_14, int($gpu14_free_bytes / ($B * 4))))")
        pc=$(( unit_params_14 - pg ))
        t14_params_gpu[$B]=$pg; t14_params_cpu[$B]=$pc
        echo "    band=${B}: params_gpu=${pg}  params_cpu=${pc}"
        if [[ "$pc" -le 0 ]]; then
            t14_cpu_gemv[$B]="0.000"
            echo "    band=${B}: all params fit in GPU — no CPU GEMV"
            continue
        fi
        F14_BENCH_CPU="${F14}/bench_cpu_b${B}.txt"
        if [[ -f "$F14_BENCH_CPU" ]] && grep -q 'Wall-time' "$F14_BENCH_CPU" 2>/dev/null; then
            echo "    (reuse $F14_BENCH_CPU)"
        else
            echo "    bench_gemv_cpu(params_cpu=${pc}, band=${B})..."
            CUDA_VISIBLE_DEVICES="$GPU" \
            "$PYTHON" "$BENCH_CPU" "$pc" "$cpu14_threads" "$B" 1 "$NUMA_CPUS" \
                2>&1 | tee "$F14_BENCH_CPU" || true
        fi
        # Wall-time avg includes GEMV + H2D (per bench design)
        t14_cpu_gemv[$B]=$(grep -oP 'Wall-time\s+avg\s+\K[0-9.]+' "$F14_BENCH_CPU" 2>/dev/null | tail -1)
        echo "    band=${B}: t_cpu_gemv=${t14_cpu_gemv[$B]:-N/A}ms"
    done

    # ─── CSV output ──────────────────────────────────────────────────────────
    # Columns:
    #   norm_gpu_total  = n_steps × (t_compute + t_gpu_gemv + t_h2d + t_noise_add) / dpsgd_epoch
    #   norm_gpu_ovhd   = n_steps × t_gpu_gemv / dpsgd_epoch          [GEMV overhead only]
    #   norm_transfer   = n_steps × t_h2d / dpsgd_epoch               [PCIe overflow]
    #   norm_cpu_total  = n_steps × max(t_compute+t_bench_gpu, t_cpu_gemv)+t_noise_add / dpsgd_epoch
    #   norm_cocoon     = (avg_epoch + build_time) / dpsgd_epoch
    csv14="${F14}/fig14.csv"
    echo "band,n_steps,fitting,params_gpu,params_cpu,norm_gpu_total,norm_gpu_ovhd,norm_transfer,norm_cpu_total,norm_cocoon" > "$csv14"
    echo ""
    printf "  %-6s %8s %15s %15s %14s\n" \
        "Band" "n_steps" "GPU-total(norm)" "CPU-total(norm)" "Cocoon(norm)"
    printf "  %-6s %8s %15s %15s %14s\n" \
        "──────" "────────" "───────────────" "───────────────" "──────────────"

    for B in "${ALL_BANDS[@]}"; do
        t_cc="${t14_cocoon[$B]:-N/A}"; t_bld="${t14_build[$B]:-0}"
        t_cpu="${t14_cpu_gemv[$B]:-N/A}"
        pg="${t14_params_gpu[$B]:-$unit_params_14}"; pc="${t14_params_cpu[$B]:-0}"
        bm1=$(( B - 1 ))
        norm_gpu_total="N/A"; norm_gpu_ovhd="N/A"; norm_xfer="N/A"
        norm_cpu_total="N/A"; norm_cc="N/A"

        if [[ "$t14_epoch" =~ ^[0-9.]+$ && "$n14_iters" =~ ^[0-9]+$ && \
              "$t14_bench_gpu" =~ ^[0-9.]+$ && \
              "$t14_compute" =~ ^[0-9.]+$ && "$t14_noise_add_val" =~ ^[0-9.]+$ ]]; then

            # Noise buffer = B vectors; overflow = bands beyond what GPU can hold
            overflow=$(( B > fitting14 ? B - fitting14 : 0 ))
            t_h2d_ms=$(python3 -c "print(f'{$overflow * $unit_params_14 * 4 / ($BW_CPUGPU_GBS * 1e9) * 1000:.3f}')")

            # t_gpu_gemv = t_bench_gpu × (unit_params / params_gpu) = t_bench_gpu × B/fitting
            if [[ "$bm1" -eq 0 ]]; then
                t_gpu_gemv_ms="0.000"
            else
                t_gpu_gemv_ms=$(python3 -c "print(f'{$t14_bench_gpu * $unit_params_14 / $pg:.3f}')")
            fi

            # GPU-GEMV bar components
            norm_gpu_total=$(python3 -c "
t = $t14_compute + $t_gpu_gemv_ms + $t_h2d_ms + $t14_noise_add_val
print(f'{$n14_iters * t / $t14_epoch:.4f}')")
            norm_gpu_ovhd=$(python3 -c "print(f'{$n14_iters * $t_gpu_gemv_ms / $t14_epoch:.4f}')")
            norm_xfer=$(python3 -c "print(f'{$n14_iters * $t_h2d_ms / $t14_epoch:.4f}')")

            # CPU-GEMV bar: t_gpu_gemv_cpu = t_bench_gpu (constant)
            if [[ "$t_cpu" =~ ^[0-9.]+$ ]]; then
                norm_cpu_total=$(python3 -c "
gpu_side = $t14_compute + $t14_bench_gpu
eff = max(gpu_side, $t_cpu) + $t14_noise_add_val
print(f'{$n14_iters * eff / $t14_epoch:.4f}')")
            fi

            [[ "$t_cc" =~ ^[0-9.]+$ ]] && \
                norm_cc=$(python3 -c "print(f'{($t_cc + $t_bld) / $t14_epoch:.4f}')")
        fi

        printf "  %-6s %8s %15s %15s %14s\n" \
            "b=$B" "${n14_iters}" "${norm_gpu_total}" "${norm_cpu_total}" "${norm_cc}"
        echo "${B},${n14_iters},${fitting14},${pg},${pc},${norm_gpu_total},${norm_gpu_ovhd},${norm_xfer},${norm_cpu_total},${norm_cc}" >> "$csv14"
    done
    echo "  → $csv14"
fi

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15a — Speedup vs band, vary d_emb (8, 16, 32; kaggle-processed, B=65536)
# ══════════════════════════════════════════════════════════════════════════════
if should_run 15a; then
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 15a — vary d_emb (8, 16, 32), kaggle-processed, B=65536"
    echo "══════════════════════════════════════════════════════════"
    F15A="${OUTPUT}/fig15a"
    mkdir -p "$F15A"

    # emb_entries for kaggle_processed ≈ 33762577 (same as Zipf1 1x)
    declare -A EMB_ROWS_A=( [8]=33762577 [16]=33762577 [32]=33762577 )
    declare -A CHUNK_SCALE_A=( [8]=1 [16]=1 [32]=1 )

    csv15a="${F15A}/fig15a.csv"
    echo "d_emb,band,t_dpsgd_epoch_ms,t_cocoon_total_ms,speedup_cocoon,speedup_gpugemv_norm" > "$csv15a"

    for d_emb in 8 16 32; do
        echo ""
        echo "  ── d_emb=${d_emb} ──"
        emb_rows="${EMB_ROWS_A[$d_emb]}"
        unit_params=$(( emb_rows * d_emb ))

        # DP-SGD baseline (reuse figure4 if d_emb=16)
        if [[ "$d_emb" -eq 16 ]]; then
            DPSGD_LOG="${SCRIPT_DIR}/figure4/dpsgd_dim16.txt"
        else
            DPSGD_LOG="${F15A}/dpsgd_emb${d_emb}.txt"
        fi
        echo "  [DP-SGD d_emb=${d_emb}]"
        run "$DPSGD_LOG" "$KAGGLE_PROC" "$d_emb" 65536 1 0
        t_dpsgd_epoch=$(avg_epoch "$DPSGD_LOG")
        n_iters=$(n_steps_from_log "$DPSGD_LOG")
        n_iters="${n_iters:-N/A}"
        echo "  t_dpsgd_epoch=${t_dpsgd_epoch:-N/A}ms  n_iters=${n_iters}"
        t_iter_ms=$(avg_iter "$DPSGD_LOG")

        GPU_BENCH_LOG="${F15A}/bench_gpu_emb${d_emb}.txt"
        CPU_BENCH_LOG="${F15A}/bench_cpu_emb${d_emb}.txt"
        for B in "${ALL_BANDS[@]}"; do
            chunk=$(chunk_for "$emb_rows" "$B")
            CLOG="${F15A}/cocoon_emb${d_emb}_b${B}.txt"
            echo "  [Cocoon d_emb=${d_emb} band=${B}]"
            run "$CLOG" "$KAGGLE_PROC" "$d_emb" 65536 "$B" "$chunk"
            t_cocoon_epoch=$(avg_epoch "$CLOG")
            t_cocoon_bld=$(total_build_ms "$CLOG"); t_cocoon_bld="${t_cocoon_bld:-0}"
            cocoon_total="N/A"
            [[ "$t_cocoon_epoch" =~ ^[0-9.]+$ ]] && \
                cocoon_total=$(python3 -c "print(f'{$t_cocoon_epoch + $t_cocoon_bld:.1f}')")

            bench_gpu_gemv "$unit_params" "$B" "$GPU_BENCH_LOG"
            bench_cpu_gemv "$unit_params" "$B" "$CPU_BENCH_LOG"
            compute_speedup "${cocoon_total:-N/A}" "${t_dpsgd_epoch:-N/A}" "${n_iters:-N/A}" "${t_iter_ms:-N/A}"

            printf "  d_emb=%-2s b=%-3s: cocoon=%8s ms  gpu_gemv=%8s ms  cpu_gemv=%8s ms  speedup=%s\n" \
                "$d_emb" "$B" "${cocoon_total:-N/A}" "${_t_gpugemv_total:-N/A}" "${_t_cpugemv_total:-N/A}" "${_speedup:-N/A}"
            echo "${d_emb},${B},${cocoon_total:-N/A},${_t_gpugemv_total:-N/A},${_t_cpugemv_total:-N/A},${_speedup:-N/A}" >> "$csv15a"
        done
    done
    echo ""
    echo "  → $csv15a"
fi

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15b — Speedup vs band, vary entries (0.5x, 1x, 2x Zipf1; d_emb=16, B=65536)
# ══════════════════════════════════════════════════════════════════════════════
if should_run 15b; then
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 15b — vary entries (0.5x,1x,2x Zipf1), d_emb=16, B=65536"
    echo "══════════════════════════════════════════════════════════"
    F15B="${OUTPUT}/fig15b"
    mkdir -p "$F15B"

    # scale_label : emb_entries : npz_path : synthetic_arg
    declare -A ENTRY_ROWS=( [0.5x]=16881289 [1x]=33762577 [2x]=67525154 )
    declare -A ENTRY_NPZ=( [0.5x]="$ZIPF1_05X" [1x]="$ZIPF1_1X" [2x]="$ZIPF1_2X" )
    declare -A ENTRY_SYN=( [0.5x]="--synthetic=half" [1x]="" [2x]="--synthetic=double" )

    csv15b="${F15B}/fig15b.csv"
    echo "scale,band,t_cocoon_total_ms,t_gpugemv_total_ms,t_cpugemv_total_ms,speedup" > "$csv15b"

    for scale in 0.5x 1x 2x; do
        echo ""
        echo "  ── entry=${scale} ──"
        emb_rows="${ENTRY_ROWS[$scale]}"
        npz="${ENTRY_NPZ[$scale]}"
        syn_arg="${ENTRY_SYN[$scale]}"
        unit_params=$(( emb_rows * 16 ))
        scale_fs="${scale//./_}"  # 0.5x → 0_5x for filenames

        # DP-SGD baseline (reuse from verify/ if 1x)
        if [[ "$scale" == "1x" ]]; then
            DPSGD_LOG="${SCRIPT_DIR}/verify/EmbEntry/Zipf1_NPP_1xE_b1_1.txt"
            [[ ! -f "$DPSGD_LOG" ]] && DPSGD_LOG="${F15B}/dpsgd_${scale_fs}.txt"
        else
            DPSGD_LOG="${F15B}/dpsgd_${scale_fs}.txt"
        fi
        echo "  [DP-SGD scale=${scale}]"
        run "$DPSGD_LOG" "$npz" 16 65536 1 0 ${syn_arg:+"$syn_arg"}
        t_dpsgd_epoch=$(avg_epoch "$DPSGD_LOG")
        n_iters=$(n_steps_from_log "$DPSGD_LOG")
        n_iters="${n_iters:-N/A}"
        echo "  t_dpsgd_epoch=${t_dpsgd_epoch:-N/A}ms  n_iters=${n_iters}"
        t_iter_ms=$(avg_iter "$DPSGD_LOG")

        GPU_BENCH_LOG="${F15B}/bench_gpu_${scale_fs}.txt"
        CPU_BENCH_LOG="${F15B}/bench_cpu_${scale_fs}.txt"
        for B in "${ALL_BANDS[@]}"; do
            chunk=$(chunk_for "$emb_rows" "$B")
            # Reuse verify/EmbEntry logs for 1x
            if [[ "$scale" == "1x" ]]; then
                CLOG="${SCRIPT_DIR}/verify/EmbEntry/Zipf1_PP_1xE_b${B}_1.txt"
                [[ ! -f "$CLOG" ]] && CLOG="${F15B}/cocoon_${scale_fs}_b${B}.txt"
            else
                CLOG="${F15B}/cocoon_${scale_fs}_b${B}.txt"
            fi
            echo "  [Cocoon scale=${scale} band=${B}]"
            run "$CLOG" "$npz" 16 65536 "$B" "$chunk" ${syn_arg:+"$syn_arg"}
            t_cocoon_epoch=$(avg_epoch "$CLOG")
            t_cocoon_bld=$(total_build_ms "$CLOG"); t_cocoon_bld="${t_cocoon_bld:-0}"
            cocoon_total="N/A"
            [[ "$t_cocoon_epoch" =~ ^[0-9.]+$ ]] && \
                cocoon_total=$(python3 -c "print(f'{$t_cocoon_epoch + $t_cocoon_bld:.1f}')")

            bench_gpu_gemv "$unit_params" "$B" "$GPU_BENCH_LOG"
            bench_cpu_gemv "$unit_params" "$B" "$CPU_BENCH_LOG"
            compute_speedup "${cocoon_total:-N/A}" "${t_dpsgd_epoch:-N/A}" "${n_iters:-N/A}" "${t_iter_ms:-N/A}"

            printf "  scale=%-4s b=%-3s: cocoon=%8s ms  gpu_gemv=%8s ms  cpu_gemv=%8s ms  speedup=%s\n" \
                "$scale" "$B" "${cocoon_total:-N/A}" "${_t_gpugemv_total:-N/A}" "${_t_cpugemv_total:-N/A}" "${_speedup:-N/A}"
            echo "${scale},${B},${cocoon_total:-N/A},${_t_gpugemv_total:-N/A},${_t_cpugemv_total:-N/A},${_speedup:-N/A}" >> "$csv15b"
        done
    done
    echo ""
    echo "  → $csv15b"
fi

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15c — Speedup vs band, vary batch (32K, 64K, 128K; d_emb=16, kaggle-processed)
# ══════════════════════════════════════════════════════════════════════════════
if should_run 15c; then
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 15c — vary batch (32K,64K,128K), d_emb=16, kaggle-processed"
    echo "══════════════════════════════════════════════════════════"
    F15C="${OUTPUT}/fig15c"
    mkdir -p "$F15C"

    declare -A BATCH_SIZES=( [32K]=32768 [64K]=65536 [128K]=131072 )
    EMB_ROWS_C=33762577
    UNIT_PARAMS_C=$(( EMB_ROWS_C * 16 ))
    BENCH_LOG_C="${F15C}/bench_gpu.txt"

    csv15c="${F15C}/fig15c.csv"
    echo "batch,band,t_cocoon_total_ms,t_gpugemv_total_ms,t_cpugemv_total_ms,speedup" > "$csv15c"

    for blabel in 32K 64K 128K; do
        batch="${BATCH_SIZES[$blabel]}"
        echo ""
        echo "  ── batch=${blabel} (${batch}) ──"

        # DP-SGD baseline (reuse figure4 for 64K)
        if [[ "$blabel" == "64K" ]]; then
            DPSGD_LOG="${SCRIPT_DIR}/figure4/dpsgd_dim16.txt"
        else
            DPSGD_LOG="${F15C}/dpsgd_${blabel}.txt"
        fi
        echo "  [DP-SGD batch=${blabel}]"
        run "$DPSGD_LOG" "$KAGGLE_PROC" 16 "$batch" 1 0
        t_dpsgd_epoch=$(avg_epoch "$DPSGD_LOG")
        n_iters=$(n_steps_from_log "$DPSGD_LOG")
        n_iters="${n_iters:-N/A}"
        echo "  t_dpsgd_epoch=${t_dpsgd_epoch:-N/A}ms  n_iters=${n_iters}"
        t_iter_ms=$(avg_iter "$DPSGD_LOG")

        CPU_BENCH_LOG_C="${F15C}/bench_cpu.txt"
        for B in "${ALL_BANDS[@]}"; do
            chunk=$(chunk_for "$EMB_ROWS_C" "$B")
            if [[ "$blabel" == "64K" ]]; then
                CLOG="${SCRIPT_DIR}/figure4/cpugemv_dim16_b${B}.txt"
                [[ ! -f "$CLOG" ]] && CLOG="${F15C}/cocoon_${blabel}_b${B}.txt"
            else
                CLOG="${F15C}/cocoon_${blabel}_b${B}.txt"
            fi
            echo "  [Cocoon batch=${blabel} band=${B}]"
            run "$CLOG" "$KAGGLE_PROC" 16 "$batch" "$B" "$chunk"
            t_cocoon_epoch=$(avg_epoch "$CLOG")
            t_cocoon_bld=$(total_build_ms "$CLOG"); t_cocoon_bld="${t_cocoon_bld:-0}"
            cocoon_total="N/A"
            [[ "$t_cocoon_epoch" =~ ^[0-9.]+$ ]] && \
                cocoon_total=$(python3 -c "print(f'{$t_cocoon_epoch + $t_cocoon_bld:.1f}')")

            bench_gpu_gemv "$UNIT_PARAMS_C" "$B" "$BENCH_LOG_C"
            bench_cpu_gemv "$UNIT_PARAMS_C" "$B" "$CPU_BENCH_LOG_C"
            compute_speedup "${cocoon_total:-N/A}" "${t_dpsgd_epoch:-N/A}" "${n_iters:-N/A}" "${t_iter_ms:-N/A}"

            printf "  batch=%-4s b=%-3s: cocoon=%8s ms  gpu_gemv=%8s ms  cpu_gemv=%8s ms  speedup=%s\n" \
                "$blabel" "$B" "${cocoon_total:-N/A}" "${_t_gpugemv_total:-N/A}" "${_t_cpugemv_total:-N/A}" "${_speedup:-N/A}"
            echo "${blabel},${B},${cocoon_total:-N/A},${_t_gpugemv_total:-N/A},${_t_cpugemv_total:-N/A},${_speedup:-N/A}" >> "$csv15c"
        done
    done
    echo ""
    echo "  → $csv15c"
fi

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15d — Speedup vs band, vary skewness (Zipf α=0.5, 1.0, 2.0; d_emb=16, B=65536)
# ══════════════════════════════════════════════════════════════════════════════
if should_run 15d; then
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo " Figure 15d — vary skewness (Zipf 0.5,1.0,2.0), d_emb=16, B=65536"
    echo "══════════════════════════════════════════════════════════"
    F15D="${OUTPUT}/fig15d"
    mkdir -p "$F15D"

    declare -A SKEW_NPZ=( [0.5]="$ZIPF05" [1.0]="$ZIPF1_1X" [2.0]="$ZIPF2" )
    EMB_ROWS_D=33762577
    UNIT_PARAMS_D=$(( EMB_ROWS_D * 16 ))
    BENCH_LOG_D="${F15D}/bench_gpu.txt"

    csv15d="${F15D}/fig15d.csv"
    echo "zipf_alpha,band,t_cocoon_total_ms,t_gpugemv_total_ms,t_cpugemv_total_ms,speedup" > "$csv15d"

    for alpha in 0.5 1.0 2.0; do
        npz="${SKEW_NPZ[$alpha]}"
        alpha_fs="${alpha//./_}"  # 0.5 → 0_5
        echo ""
        echo "  ── Zipf α=${alpha} ──"

        # DP-SGD baseline (reuse Fig 14 for Zipf1.0=1x)
        if [[ "$alpha" == "1.0" ]]; then
            DPSGD_LOG="${SCRIPT_DIR}/verify/EmbEntry/Zipf1_NPP_1xE_b1_1.txt"
            [[ ! -f "$DPSGD_LOG" ]] && DPSGD_LOG="${F15D}/dpsgd_zipf${alpha_fs}.txt"
        else
            DPSGD_LOG="${F15D}/dpsgd_zipf${alpha_fs}.txt"
        fi
        echo "  [DP-SGD Zipf α=${alpha}]"
        run "$DPSGD_LOG" "$npz" 16 65536 1 0
        t_dpsgd_epoch=$(avg_epoch "$DPSGD_LOG")
        n_iters=$(n_steps_from_log "$DPSGD_LOG")
        n_iters="${n_iters:-N/A}"
        t_iter_ms=$(avg_iter "$DPSGD_LOG")
        echo "  t_dpsgd_epoch=${t_dpsgd_epoch:-N/A}ms  n_iters=${n_iters}"

        GPU_BENCH_LOG="${F15D}/bench_gpu_zipf${alpha_fs}.txt"
        CPU_BENCH_LOG="${F15D}/bench_cpu_zipf${alpha_fs}.txt"
        for B in "${ALL_BANDS[@]}"; do
            chunk=$(chunk_for "$EMB_ROWS_D" "$B")
            if [[ "$alpha" == "1.0" ]]; then
                CLOG="${SCRIPT_DIR}/verify/EmbEntry/Zipf1_PP_1xE_b${B}_1.txt"
                [[ ! -f "$CLOG" ]] && CLOG="${F15D}/cocoon_zipf${alpha_fs}_b${B}.txt"
            else
                CLOG="${F15D}/cocoon_zipf${alpha_fs}_b${B}.txt"
            fi
            echo "  [Cocoon Zipf α=${alpha} band=${B}]"
            run "$CLOG" "$npz" 16 65536 "$B" "$chunk"
            t_cocoon_epoch=$(avg_epoch "$CLOG")
            t_cocoon_bld=$(total_build_ms "$CLOG"); t_cocoon_bld="${t_cocoon_bld:-0}"
            cocoon_total="N/A"
            [[ "$t_cocoon_epoch" =~ ^[0-9.]+$ ]] && \
                cocoon_total=$(python3 -c "print(f'{$t_cocoon_epoch + $t_cocoon_bld:.1f}')")

            bench_gpu_gemv "$UNIT_PARAMS_D" "$B" "$GPU_BENCH_LOG"
            bench_cpu_gemv "$UNIT_PARAMS_D" "$B" "$CPU_BENCH_LOG"
            compute_speedup "${cocoon_total:-N/A}" "${t_dpsgd_epoch:-N/A}" "${n_iters:-N/A}" "${t_iter_ms:-N/A}"

            printf "  α=%-4s b=%-3s: cocoon=%8s ms  gpu_gemv=%8s ms  cpu_gemv=%8s ms  speedup=%s\n" \
                "$alpha" "$B" "${cocoon_total:-N/A}" "${_t_gpugemv_total:-N/A}" "${_t_cpugemv_total:-N/A}" "${_speedup:-N/A}"
            echo "${alpha},${B},${cocoon_total:-N/A},${_t_gpugemv_total:-N/A},${_t_cpugemv_total:-N/A},${_speedup:-N/A}" >> "$csv15d"
        done
    done
    echo ""
    echo "  → $csv15d"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " All CSVs saved to ${OUTPUT}/"
echo "════════════════════════════════════════════════════════════════"
