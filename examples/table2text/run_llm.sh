#!/usr/bin/env bash
# Figure 18: End-to-end normalized training time — Cocoon+NMP vs GPU-GEMV vs CPU-GEMV.
#
# Models: OPT-350M (band=256), GPT2-L (band=112), OPT-1.3B (band=64), GPT2-XL (band=56)
#
# Bar components (all normalized to t_gpu = pure GPU compute per step):
#   Cocoon   : max(Train(GPU)=t_gpu,  GEMV(CXL)=NMP/step,  GEMV(CPU)=GEMV/step)
#   GPU-GEMV : Train(GPU)=t_gpu  +  Transfer(MainMem)=cpu_data/BW_CPUGPU
#                                +  Transfer(CXL)=cxl_data/BW_CXLGPU       
#   CPU-GEMV : max(Train(GPU)=t_gpu,  Transfer(CXL)+GEMV(CPU)=cxl_data/BW_CXLCPU + cocoon_cpu_gemv*(CPU+NMP)/CPU)
#
# Usage:
#   bash run_llm.sh                        # all 4 models
#   bash run_llm.sh opt-350m              # one model only
#
# Profile phase uses profile_hardware.sh (fixed) to measure GPU_FREE, CPU_FREE,
# physical batch size, and trainable param count.  Results are cached in
# figure18/<model>/profile/ so re-runs skip profiling automatically.
#
# Partition selection:
#   n_part = ceil((band-1)*params_per_rank*4 / (gpu_free_bytes * SAFETY))
#   C_part = floor((cpu_free_bytes - CPU_SAFETY_BYTES) / partition_noise_bytes)
#   NMP_part = n_part - 1 - C_part   (G_part always = 1)
#
# Retry logic:
#   GPU OOM      → n_part += 1
#   CPU memory   → background monitor kills training when NUMA free < KILL_THRESHOLD_GB
#                  then C_part -= 1 and retry

set -uo pipefail

# Kill entire process tree on Ctrl+C or script exit with a live training job.
_TRAIN_PID=""
_cleanup() {
    if [[ -n "$_TRAIN_PID" ]]; then
        kill -TERM -"$_TRAIN_PID" 2>/dev/null || kill -TERM "$_TRAIN_PID" 2>/dev/null || true
    fi
}
trap '_cleanup' INT TERM EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_HW="${SCRIPT_DIR}/profile_hardware.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_ZERO1.sh"
OUTPUT="${SCRIPT_DIR}/figure18"

# ══════════════════════════════════════════════════════════════════════════════
# USER-ADJUSTABLE
# ══════════════════════════════════════════════════════════════════════════════
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface}"

NUM_GPUS="${NUM_GPUS:-4}"
GPUS="${GPUS:-0,1,2,3}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/prefix-tuning}"
MASTER_PORT="${MASTER_PORT:-29700}"
TASK="e2e"
TARGET_EPSILON=8
LOGICAL_BATCH=1024
MAX_STEPS=10           # steps for the Cocoon timing run
SAFETY=0.80            # GPU memory safety fraction for n_part computation
CPU_SAFETY_BYTES=$(( 5 * 1000000000 ))   # 5 GB additive headroom for CPU
CXL_PER_RANK_GB=64     # 256 GB CXL device / 4 GPUs
KILL_THRESHOLD_GB=4    # kill training if NUMA node free memory drops below this

# Bandwidth (bytes/s)
BW_CPUGPU_BPS=23300000000   # PCIe 4.0 per GPU link
BW_CXLGPU_BPS=5520000000   # CXL shared bus / 4 GPU slots
BW_CXLCPU_BPS=5600000000   # CXL→CPU shared bus / 4 slots

# (model-key) → "hf_name:band"
declare -A MODEL_CFG=(
    [opt-350m]="facebook/opt-350m:256"
    [gpt2-large]="gpt2-large:112"
    [opt-1.3b]="facebook/opt-1.3b:64"
    [gpt2-xl]="gpt2-xl:56"
    [opt-2.7b]="facebook/opt-2.7b:32"
)
MODEL_ORDER=(opt-350m gpt2-large opt-1.3b gpt2-xl opt-2.7b)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER MODELS (optional first arg)
# ══════════════════════════════════════════════════════════════════════════════
if [[ $# -ge 1 ]]; then
    MODEL_ORDER=("$1")
fi

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

numa_avail_kb() {
    awk '/MemFree|FilePages|SReclaimable/{s+=$4} END{print s+0}' \
        "/sys/devices/system/node/node${1}/meminfo" 2>/dev/null || echo 0
}

# Parse a key=value line from profile_hardware.sh summary output
parse_profile_key() {
    local key=$1 logfile=$2
    grep -oP "${key}=\K[0-9.]+" "$logfile" 2>/dev/null | tail -1
}

# Compute pure GPU training time per step: (forward + loss + backward + clip) * grad_accum.
# Excludes optim_step, which stalls on the noise queue when CXL/CPU is the bottleneck.
# Takes max across all device prints for each component (bottleneck rank).
extract_t_gpu_ms() {
    local log=$1 grad_accum=$2
    python3 -c "
import re, sys
try:
    log = open('$log').read()
except Exception as e:
    sys.exit(0)
def max_val(label):
    vals = re.findall(r'^' + label + r'\s*:\s*([0-9.]+)', log, re.MULTILINE)
    return max((float(v) for v in vals), default=0.0)
fwd  = max_val('forward')
loss = max_val('loss')
bwd  = max_val('backward')
clip = max_val('clip')
if fwd == 0.0:
    sys.exit(0)
print(f'{(fwd + loss + bwd + clip) * $grad_accum:.3f}')
" 2>/dev/null
}

# Parse NMP×N total/step (max across all device prints = bottleneck rank)
extract_nmp_ms() {
    grep -oP 'NMP×[0-9]+: total/step=\K[0-9.]+' "$1" 2>/dev/null | sort -n | tail -1
}

# Parse noise_worker FINAL gemv time (max across workers = bottleneck)
extract_worker_gemv_ms() {
    grep -oP '\[noise_worker rank=[0-9]+\] FINAL:.*\(≈\K[0-9]+(?=ms gemv)' \
        "$1" 2>/dev/null | sort -n | tail -1
}

# Parse noise_worker FINAL total/step (max across workers)
extract_worker_total_ms() {
    grep -oP '\[noise_worker rank=[0-9]+\] FINAL:.*total/step=\K[0-9.]+' \
        "$1" 2>/dev/null | sort -n | tail -1
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Profile hardware for one model
# Sets globals: _gpu_free_gb  _cpu_free_gb  _found_bs  _params  _active_numa
# Caches results in <model_dir>/profile/profile_summary.txt
# ══════════════════════════════════════════════════════════════════════════════
profile_model() {
    local model_hf=$1 model_dir=$2
    local prof_dir="${model_dir}/profile"
    local summary="${prof_dir}/profile_summary.txt"
    local prof_log="${prof_dir}/profile_hardware.log"
    mkdir -p "$prof_dir"

    # Check if summary has content (not just exists — a prior failed run may leave an empty file)
    local summary_ok=0
    [[ -f "$summary" ]] && grep -q 'GPU_FREE_GB' "$summary" 2>/dev/null && summary_ok=1

    if [[ $summary_ok -eq 1 ]]; then
        echo "  (reusing profile: $summary)"
    elif [[ -f "$prof_log" ]] && grep -q 'Number of trainable params' "$prof_log" 2>/dev/null; then
        # Prior run completed training but profile_hardware.sh exited before Phase 4.
        # Recover values directly from the existing log.
        echo "  (recovering profile from existing log: $prof_log)"
        local gpu_total_mib gpu_peak_mib phys_bs numa_baseline_gb cpu_used_peak_gb gpu_free_gb cpu_free_gb
        gpu_total_mib=$(grep -oP 'GPU: total=\K[0-9]+' "$prof_log" | tail -1)
        gpu_peak_mib=$(grep -oP 'torch\.cuda\.max_memory_allocated:\s*\K[0-9.]+' "$prof_log" \
          | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{printf "%d\n", m}')
        phys_bs=$(grep -oP 'attempt [0-9]+: physical_bs=\K[0-9]+(?= \.\.\. OK)' "$prof_log" | tail -1)
        # NUMA total RAM (hardware constant; read from current system)
        local active_numa_node
        active_numa_node=$(grep -oP 'NUMA node\K[0-9]+' "$prof_log" | tail -1)
        active_numa_node="${active_numa_node:-1}"
        local numa_total_kb
        numa_total_kb=$(awk '/MemTotal/{print $4}' \
          "/sys/devices/system/node/node${active_numa_node}/meminfo" 2>/dev/null || echo 0)
        local numa_total_gb
        numa_total_gb=$(python3 -c "print(f'{${numa_total_kb:-0}/1024/1024:.2f}')")
        # Peak system-wide committed CPU RAM during training (psutil total-available)
        cpu_used_peak_gb=$(grep -oP 'CPU Virtual Memory:\s*used\s*=\s*\K[0-9.]+' "$prof_log" \
          | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m}')
        gpu_free_gb=$(python3 -c "print(f'{(${gpu_total_mib:-0} - ${gpu_peak_mib:-0})/1024:.2f}')")
        cpu_free_gb=$(python3 -c "print(f'{max(0.0, ${numa_total_gb:-0} - ${cpu_used_peak_gb:-0}):.2f}')")
        {
            echo "GPU_FREE_GB=${gpu_free_gb}"
            echo "CPU_FREE_GB=${cpu_free_gb}"
            echo "PHYSICAL_BS=${phys_bs:-1}"
            echo "PARAMS=$(grep -oP 'Number of trainable params:\s*\K[0-9]+' "$prof_log" | tail -1)"
            echo "ACTIVE_NUMA=${active_numa_node}"
        } > "$summary"
        echo "  Recovered: GPU_FREE=${gpu_free_gb}GB  CPU_FREE=${cpu_free_gb}GB (node${active_numa_node}_total=${numa_total_gb}GB - cpu_peak=${cpu_used_peak_gb}GB)  BS=${phys_bs}"
    else
        echo "  Running profile_hardware.sh for ${model_hf}..."
        bash "$PROFILE_HW" \
            --train-script   "$TRAIN_SCRIPT" \
            --data-dir       "$DATA_DIR" \
            --output-dir     "$prof_dir/tmp" \
            --model          "$model_hf" \
            --logical-batch  "$LOGICAL_BATCH" \
            --num-gpus       "$NUM_GPUS" \
            --gpus           "$GPUS" \
            --port           "$MASTER_PORT" \
            --epsilon        "$TARGET_EPSILON" \
            2>&1 | tee "$prof_log"

        # Write compact summary for caching
        {
            grep -oP 'GPU_FREE_GB=[0-9.]+' "$prof_log" | tail -1
            grep -oP 'CPU_FREE_GB=[0-9.]+' "$prof_log" | tail -1
            grep -oP 'PHYSICAL_BS=[0-9]+'  "$prof_log" | tail -1
            grep -oP 'PARAMS=[0-9]+'       "$prof_log" | tail -1
        } > "$summary"
        grep -oP 'Active NUMA node:\s*node\K[0-9]+' "$prof_log" | tail -1 \
            | xargs -I{} echo "ACTIVE_NUMA={}" >> "$summary" || true
    fi

    _gpu_free_gb=$(parse_profile_key GPU_FREE_GB "$summary")
    _cpu_free_gb=$(parse_profile_key CPU_FREE_GB "$summary")
    _found_bs=$(parse_profile_key    PHYSICAL_BS "$summary")
    _params=$(parse_profile_key      PARAMS      "$summary")
    _active_numa=$(grep -oP 'ACTIVE_NUMA=\K[0-9]+' "$summary" 2>/dev/null | tail -1)
    _active_numa="${_active_numa:-1}"   # default to node 1 if not found

    echo "  GPU_FREE=${_gpu_free_gb}GB  CPU_FREE=${_cpu_free_gb}GB  BS=${_found_bs}  PARAMS=${_params}  NUMA=${_active_numa}"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Compute partitions
# Sets globals: _n_part  _c_part  _nmp_part  _partition_noise_bytes
# ══════════════════════════════════════════════════════════════════════════════
compute_partitions() {
    local params_per_rank=$1 band=$2 gpu_free_bytes=$3 cpu_free_bytes=$4
    local safety_override=${5:-$SAFETY}

    local cxl_bytes total_noise_bytes
    cxl_bytes=$(python3 -c "print(int($CXL_PER_RANK_GB * 1e9))")
    total_noise_bytes=$(python3 -c "print(($band - 1) * $params_per_rank * 4)")

    read _n_part _partition_noise_bytes _c_part _nmp_part < <(python3 -c "
import math, sys
total_noise = $total_noise_bytes
gpu_free    = $gpu_free_bytes
cpu_free    = $cpu_free_bytes
cxl_bytes   = $cxl_bytes
safety      = $safety_override
cpu_safety  = $CPU_SAFETY_BYTES
num_gpus    = $NUM_GPUS

gpu_cap      = gpu_free * safety
cpu_per_rank = max(0.0, cpu_free - cpu_safety) / num_gpus

# Check at minimum n_part whether integer solution exists
n_part    = max(1, math.ceil(total_noise / gpu_cap))
part_bytes = total_noise / n_part
max_c_0   = math.floor(cpu_per_rank / part_bytes)
max_nmp_0 = math.floor(cxl_bytes    / part_bytes)

if max_c_0 + max_nmp_0 < n_part - 1:
    # Integer partition constraint fails at minimum n_part with cpu_safety.
    # Iterate n_part upward using cpu_free (no safety margin) until valid.
    found = False
    for n in range(n_part, n_part + 100):
        pb   = total_noise / n
        c_mx = math.floor(cpu_free / (num_gpus * pb))
        c_mx = min(c_mx, n - 1)
        nmp  = math.floor(cxl_bytes / pb)
        nmp  = min(nmp, n - 1)
        c    = n - 1 - nmp
        if c <= c_mx:
            overage = max(0.0, c * num_gpus * pb - (cpu_free - cpu_safety)) / 1e9
            if overage > 0:
                print(f'WARNING: cpu_safety relaxed by {overage:.1f} GB (n_part={n}, C={c}, NMP={nmp})', file=sys.stderr)
            print(n, int(pb), c, nmp)
            found = True
            break
    if not found:
        print(f'INFEASIBLE: band requires {total_noise/1e9:.1f} GB noise/rank, '
              f'no valid partition found within 100 iterations', file=sys.stderr)
        sys.exit(1)
    sys.exit(0)

# Normal solver — converges in first iteration for feasible configs
for _ in range(50):
    part_bytes = total_noise / n_part
    max_c    = max(0, math.floor((cpu_free - cpu_safety) / (num_gpus * part_bytes)))
    max_c    = min(max_c, n_part - 1)
    max_nmp  = max(0, math.floor(cxl_bytes / part_bytes))
    c_part   = max_c
    nmp_part = n_part - 1 - c_part
    if nmp_part <= max_nmp:
        break
    nmp_part = max_nmp
    c_part   = n_part - 1 - nmp_part
    if c_part * num_gpus * part_bytes <= cpu_free - cpu_safety:
        break
    n_part += 1

print(n_part, int(part_bytes), c_part, nmp_part)
")
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Run Cocoon with retry
# On GPU OOM: n_part += 1, recompute
# On CPU memory pressure (kill by monitor): c_part -= 1, retry
# Sets globals: _cocoon_log  _final_n_part  _final_c_part  _final_nmp_part
# ══════════════════════════════════════════════════════════════════════════════
run_cocoon_with_retry() {
    local model_hf=$1 model_dir=$2 band=$3 found_bs=$4
    local params_per_rank=$5 gpu_free_bytes=$6 cpu_free_bytes=$7 active_numa=$8

    local kill_threshold_kb=$(( KILL_THRESHOLD_GB * 1024 * 1024 ))
    local cur_n_part=$_n_part cur_c_part=$_c_part cur_nmp_part=$_nmp_part
    local success=0 port_offset=0

    echo "  Initial partitions: n_part=${cur_n_part} G=1 C=${cur_c_part} NMP=${cur_nmp_part}"

    for retry in $(seq 0 9); do
        _cocoon_log="${model_dir}/cocoon_np${cur_n_part}_g1_c${cur_c_part}_b${band}.txt"

        if [[ -f "$_cocoon_log" ]] && grep -q 'step_by_step' "$_cocoon_log" 2>/dev/null; then
            echo "  (reusing $_cocoon_log)"
            success=1; break
        fi

        echo "  [retry=${retry}] n_part=${cur_n_part} G=1 C=${cur_c_part} NMP=${cur_nmp_part}"

        # ── Background NUMA kill monitor ─────────────────────────────────────
        local monitor_flag_file
        monitor_flag_file=$(mktemp)
        (
            while true; do
                local avail_kb
                avail_kb=$(numa_avail_kb "$active_numa")
                if (( avail_kb < kill_threshold_kb )); then
                    echo "cpu_pressure" > "$monitor_flag_file"
                    # Kill the entire process group launched below
                    [[ -f "${monitor_flag_file}.pgid" ]] && \
                        kill -TERM -"$(cat "${monitor_flag_file}.pgid")" 2>/dev/null || true
                    break
                fi
                sleep 1
            done
        ) &
        local monitor_pid=$!

        # ── Launch training in its own process group ─────────────────────────
        set +e
        CUDA_VISIBLE_DEVICES="$GPUS" setsid bash "$TRAIN_SCRIPT" \
            "$DATA_DIR" "$OUTPUT/tmp" "$TASK" "$model_hf" \
            "$found_bs" "$LOGICAL_BATCH" \
            "no" "$band" "$NUM_GPUS" "$(( MASTER_PORT + 200 + retry + port_offset ))" \
            "$cur_n_part" "1" "$cur_c_part" \
            "MixGhostClip" "$TARGET_EPSILON" "automatic" "all-layer" \
            "no" "no" "no" "no" \
            "$MAX_STEPS" "True" \
            > "$_cocoon_log" 2>&1 &
        local train_pid=$!
        _TRAIN_PID=$train_pid
        echo "$train_pid" > "${monitor_flag_file}.pgid"
        wait "$train_pid"
        local exit_code=$?
        _TRAIN_PID=""
        set -e

        kill "$monitor_pid" 2>/dev/null; wait "$monitor_pid" 2>/dev/null || true

        # ── Classify outcome ─────────────────────────────────────────────────
        if grep -q 'step_by_step' "$_cocoon_log" 2>/dev/null; then
            success=1; break
        elif [[ "$(cat "$monitor_flag_file" 2>/dev/null)" == "cpu_pressure" ]]; then
            echo "  → CPU memory pressure; decreasing C_part"
            cur_c_part=$(( cur_c_part - 1 ))
            if (( cur_c_part < 0 )); then
                echo "  ERROR: C_part < 0; cannot reduce further."
                rm -f "$monitor_flag_file" "${monitor_flag_file}.pgid"
                break
            fi
            cur_nmp_part=$(( cur_n_part - 1 - cur_c_part ))
            rm -f "$_cocoon_log"
        elif grep -qiE "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" \
                "$_cocoon_log" 2>/dev/null; then
            echo "  → GPU OOM; increasing n_part"
            cur_n_part=$(( cur_n_part + 1 ))
            local cxl_bytes_retry
            cxl_bytes_retry=$(python3 -c "print(int($CXL_PER_RANK_GB * 1e9))")
            read cur_n_part _pb cur_c_part cur_nmp_part < <(python3 -c "
import math, sys
total_noise = ($band - 1) * $params_per_rank * 4
gpu_free    = $gpu_free_bytes
cpu_free    = $cpu_free_bytes
cxl_bytes   = $cxl_bytes_retry
safety      = $SAFETY
cpu_safety  = $CPU_SAFETY_BYTES
num_gpus    = $NUM_GPUS
n_part      = $cur_n_part
gpu_cap     = gpu_free * safety
cpu_per_rank = max(0.0, cpu_free - cpu_safety) / num_gpus
for _ in range(50):
    part_bytes = total_noise / n_part
    max_c    = max(0, math.floor((cpu_free - cpu_safety) / (num_gpus * part_bytes)))
    max_c    = min(max_c, n_part - 1)
    max_nmp  = max(0, math.floor(cxl_bytes / part_bytes))
    c_part   = max_c
    nmp_part = n_part - 1 - c_part
    if nmp_part <= max_nmp:
        break
    nmp_part = max_nmp
    c_part   = n_part - 1 - nmp_part
    if c_part * num_gpus * part_bytes <= cpu_free - cpu_safety:
        break
    n_part += 1
print(n_part, int(part_bytes), c_part, nmp_part)
")
            rm -f "$_cocoon_log"
            port_offset=$(( port_offset + 10 ))
        else
            echo "  → Unknown failure (exit=${exit_code}); stopping retries."
            echo "  → Log saved: $_cocoon_log"
            rm -f "$monitor_flag_file" "${monitor_flag_file}.pgid"
            break
        fi
        rm -f "$monitor_flag_file" "${monitor_flag_file}.pgid"
    done

    _final_n_part=$cur_n_part
    _final_c_part=$cur_c_part
    _final_nmp_part=$cur_nmp_part

    if [[ $success -eq 0 ]]; then
        echo "  ERROR: Cocoon run failed for ${model_hf} after retries."
        return 1
    fi
    return 0
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
mkdir -p "$OUTPUT"

CSV="${OUTPUT}/figure18.csv"
echo "model,band,n_part,g_part,c_part,nmp_part,\
params_total,params_per_rank,\
t_gpu_ms,nmp_ms,worker_gemv_ms,worker_total_ms,\
cpu_data_gb,cxl_data_gb,\
norm_cocoon_train,norm_cocoon_cxl,norm_cocoon_cpu,norm_cocoon_total,\
norm_gpugemv_train,norm_gpugemv_xfer_cpu,norm_gpugemv_xfer_cxl,norm_gpugemv_total,\
norm_cpugemv_train,norm_cpugemv_cpu,norm_cpugemv_total" > "$CSV"

for model_key in "${MODEL_ORDER[@]}"; do
    if [[ -z "${MODEL_CFG[$model_key]+x}" ]]; then
        echo "Unknown model key: $model_key  (choices: ${!MODEL_CFG[*]})"
        exit 1
    fi
    IFS=':' read -r model_hf band <<< "${MODEL_CFG[$model_key]}"
    model_dir="${OUTPUT}/${model_key}"
    mkdir -p "$model_dir"

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo " ${model_key}  (${model_hf})  band=${band}"
    echo "══════════════════════════════════════════════════════════════════════"

    # ── Profile ──────────────────────────────────────────────────────────────
    echo ""
    echo "[1/3] Hardware profiling"
    profile_model "$model_hf" "$model_dir"

    params_total="$_params"
    gpu_free_gb="$_gpu_free_gb"
    cpu_free_gb="$_cpu_free_gb"
    found_bs="$_found_bs"
    active_numa="$_active_numa"

    if [[ ! "$params_total" =~ ^[0-9]+$ || "$params_total" -eq 0 ]]; then
        echo "  ERROR: Could not determine param count; skipping ${model_key}."
        continue
    fi

    params_per_rank=$(python3 -c "print($params_total // $NUM_GPUS)")
    grad_accum=$(python3 -c "print($LOGICAL_BATCH // ($NUM_GPUS * $found_bs))")
    gpu_free_bytes=$(python3 -c "print(int($gpu_free_gb * 1e9))")
    cpu_free_bytes=$(python3 -c "print(int($cpu_free_gb * 1e9))")
    cxl_bytes=$(python3 -c "print(int($CXL_PER_RANK_GB * 1e9))")

    # ── Band validation ───────────────────────────────────────────────────────
    max_safe_band=$(python3 -c "
per_slice = $params_per_rank * 4
total = $gpu_free_bytes + $cpu_free_bytes + $cxl_bytes
import math; print(int(total * 0.95 / per_slice))
")
    echo "  max_safe_band=${max_safe_band}  requested_band=${band}"
    if (( band > max_safe_band )); then
        echo "  WARNING: band=${band} exceeds 0.95 × available memory (max=${max_safe_band})."
        echo "  Skipping ${model_key}."
        continue
    fi

    # ── Partitions ────────────────────────────────────────────────────────────
    echo ""
    echo "[2/3] Partition selection + Cocoon run"
    # Use a more lenient safety fraction when GPU free space is very tight (<2 GB),
    # so the noise partition stays small enough to avoid OOM during training.
    local_safety=$(python3 -c "print(0.57 if $gpu_free_gb < 2.0 else $SAFETY)")
    (( $(python3 -c "print(1 if $gpu_free_gb < 2.0 else 0)") )) && \
        echo "  (GPU free=${gpu_free_gb} GB < 2 GB: using safety=${local_safety} instead of ${SAFETY})"
    compute_partitions "$params_per_rank" "$band" "$gpu_free_bytes" "$cpu_free_bytes" "$local_safety"
    echo "  Computed: n_part=${_n_part} G=1 C=${_c_part} NMP=${_nmp_part}"
    echo "  partition_noise_bytes=${_partition_noise_bytes} ($(python3 -c "print(f'{$_partition_noise_bytes/1e9:.2f}')") GB)"

    # ── Cocoon run ────────────────────────────────────────────────────────────
    if ! run_cocoon_with_retry \
            "$model_hf" "$model_dir" "$band" "$found_bs" \
            "$params_per_rank" "$gpu_free_bytes" "$cpu_free_bytes" "$active_numa"; then
        continue
    fi

    n_part="$_final_n_part"
    c_part="$_final_c_part"
    nmp_part="$_final_nmp_part"
    cocoon_log="$_cocoon_log"
    echo "  Final partitions: n_part=${n_part} G=1 C=${c_part} NMP=${nmp_part}"

    # ── Parse timing ──────────────────────────────────────────────────────────
    echo ""
    echo "[3/3] Computing bar values"

    step_ms=$(extract_t_gpu_ms  "$cocoon_log" "$grad_accum")
    nmp_ms=$(extract_nmp_ms     "$cocoon_log")
    worker_gemv_ms=$(extract_worker_gemv_ms   "$cocoon_log")
    worker_total_ms=$(extract_worker_total_ms "$cocoon_log")

    if [[ ! "$step_ms" =~ ^[0-9.]+$ ]]; then
        echo "  ERROR: Could not parse forward/loss/backward/clip from $cocoon_log"
        continue
    fi

    nmp_ms="${nmp_ms:-0}"
    worker_gemv_ms="${worker_gemv_ms:-0}"
    worker_total_ms="${worker_total_ms:-0}"

    worker_transfer_ms=$(python3 -c "
total=${worker_total_ms}; gemv=${worker_gemv_ms}
print(f'{max(0.0, total - gemv):.2f}')
")

    # Data sizes per rank held in CPU and NMP memory (bytes)
    cpu_data_bytes=$(python3 -c "
print(int($c_part * ($band-1) * $params_per_rank * 4 / $n_part))
")
    cxl_data_bytes=$(python3 -c "
print(int($nmp_part * ($band-1) * $params_per_rank * 4 / $n_part))
")
    cpu_data_gb=$(python3 -c "print(f'{$cpu_data_bytes/1e9:.3f}')")
    cxl_data_gb=$(python3 -c "print(f'{$cxl_data_bytes/1e9:.3f}')")

    # Analytical transfer times (ms)
    t_xfer_cpu_gpu=$(python3 -c "print(f'{$cpu_data_bytes/$BW_CPUGPU_BPS*1000:.3f}')")
    t_xfer_cxl_gpu=$(python3 -c "print(f'{$cxl_data_bytes/$BW_CXLGPU_BPS*1000:.3f}')")
    t_xfer_cxl_cpu=$(python3 -c "print(f'{$cxl_data_bytes/$BW_CXLCPU_BPS*1000:.3f}')")

    # CPU-GEMV: scale measured CPU GEMV by (C_part + NMP_part) / C_part
    if [[ "$c_part" -gt 0 && "$worker_gemv_ms" =~ ^[0-9.]+$ ]]; then
        t_cpu_gemv_scaled=$(python3 -c "
print(f'{$worker_gemv_ms * ($c_part + $nmp_part) / $c_part:.3f}')
")
    else
        t_cpu_gemv_scaled="$worker_gemv_ms"
    fi

    # Normalise by t_gpu (pure GPU compute per step).
    # Cocoon   : 3 parallel bars [Train | CXL GEMV | CPU GEMV]  → total = max
    # CPU-GEMV : 2 parallel bars [Train | CXL→CPU xfer + CPU GEMV] → total = max
    # GPU-GEMV : 1 serial bar    [Train + CPU→GPU xfer + CXL→GPU xfer] → total = sum
    python3 -c "
s   = $step_ms
nmp = $nmp_ms
wt  = $worker_total_ms
tc  = $t_xfer_cpu_gpu
tg  = $t_xfer_cxl_gpu
tk  = $t_xfer_cxl_cpu
gc  = $t_cpu_gemv_scaled
norm = lambda x: f'{x/s:.4f}'
cocoon_total   = max(s, nmp, wt) / s
gpugemv_total  = (s + tc + tg) / s
cpugemv_cpu    = tk + gc
cpugemv_total  = max(s, cpugemv_cpu) / s
print()
print('  Normalised (t_gpu={:.1f}ms = (fwd+loss+bwd+clip)×grad_accum):'.format(s))
print('  Cocoon   : [Train={} | CXL={} | CPU={}]  total={:.4f}'.format(norm(s), norm(nmp), norm(wt), cocoon_total))
print('  GPU-GEMV : Train+Xfer(CPU)+Xfer(CXL) = {:.4f}'.format(gpugemv_total))
print('  CPU-GEMV : [Train={} | Xfer(CXL)+GEMV(CPU)={}]  total={:.4f}'.format(norm(s), norm(cpugemv_cpu), cpugemv_total))
"
    # CSV row
    python3 -c "
s   = $step_ms
nmp = $nmp_ms
wt  = $worker_total_ms
tc  = $t_xfer_cpu_gpu; tg = $t_xfer_cxl_gpu
tk  = $t_xfer_cxl_cpu; gc = $t_cpu_gemv_scaled
norm = lambda x: f'{x/s:.4f}'
cocoon_total  = f'{max(s,nmp,wt)/s:.4f}'
gpugemv_total = f'{(s+tc+tg)/s:.4f}'
cpugemv_total = f'{max(s,tk+gc)/s:.4f}'
print('${model_key},${band},${n_part},1,${c_part},${nmp_part},'
      '${params_total},${params_per_rank},'
      '${step_ms},${nmp_ms},${worker_gemv_ms},${worker_total_ms},'
      '${cpu_data_gb},${cxl_data_gb},'
      f'{norm(s)},{norm(nmp)},{norm(wt)},{cocoon_total},'
      f'{norm(s)},{norm(tc)},{norm(tg)},{gpugemv_total},'
      f'{norm(s)},{norm(tk+gc)},{cpugemv_total}')
" >> "$CSV"

    echo "  → $CSV"
done

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Figure 18 data saved to ${CSV}"
echo "════════════════════════════════════════════════════════════════════════"
