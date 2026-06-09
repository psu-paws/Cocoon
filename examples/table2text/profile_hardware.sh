#!/usr/bin/env bash
# profile_hardware.sh — measure GPU/CPU memory peak and step latency for partition selection.
# Called by examples/table2text/run_llm.sh.

set -uo pipefail

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface}"

TRAIN_SCRIPT=""; DATA_DIR=""; OUTPUT_DIR=""
MODEL="gpt2-medium"; LOGICAL_BATCH=1024; NUM_GPUS=4; GPUS="4,5,6,7"
NUMACTL_ARGS=""; NUMA_NODE="auto"; TASK="e2e"
PORT=29500; TARGET_EPSILON=8; PROFILE_STEPS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-script)    TRAIN_SCRIPT="$2";    shift 2 ;;
    --data-dir)        DATA_DIR="$2";        shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
    --model)           MODEL="$2";           shift 2 ;;
    --logical-batch)   LOGICAL_BATCH="$2";   shift 2 ;;
    --num-gpus)        NUM_GPUS="$2";        shift 2 ;;
    --gpus)            GPUS="$2";            shift 2 ;;
    --numactl-args)    NUMACTL_ARGS="$2";    shift 2 ;;
    --numa-node)       NUMA_NODE="$2";       shift 2 ;;
    --task)            TASK="$2";            shift 2 ;;
    --port)            PORT="$2";            shift 2 ;;
    --epsilon)         TARGET_EPSILON="$2";  shift 2 ;;
    --profile-steps)   PROFILE_STEPS="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

for req in TRAIN_SCRIPT DATA_DIR OUTPUT_DIR; do
  [[ -z "${!req}" ]] && { echo "Error: --${req//_/-} is required."; exit 1; }
done
[[ ! -f "$TRAIN_SCRIPT" ]] && { echo "Error: train script not found: $TRAIN_SCRIPT"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$TRAIN_SCRIPT")" && pwd)"
mkdir -p "$OUTPUT_DIR"

TMPDIR_PROF=$(mktemp -d)
NUMA_SAMPLE_FILE="$TMPDIR_PROF/numa_samples.txt"
TRAIN_LOG="$TMPDIR_PROF/train_profile.log"

cleanup() {
  [[ -n "${SAMPLER_PID:-}" ]] && { kill "$SAMPLER_PID" 2>/dev/null || true; }
  rm -rf "$TMPDIR_PROF"
}
trap cleanup EXIT

if [[ -n "$NUMACTL_ARGS" ]] && command -v numactl &>/dev/null; then
  NUMACTL_PREFIX="numactl $NUMACTL_ARGS"
elif [[ -n "$NUMACTL_ARGS" ]]; then
  echo "WARNING: --numactl-args given but numactl not found; running without NUMA binding."
  NUMACTL_PREFIX=""
else
  NUMACTL_PREFIX=""
fi

numa_avail_kb() {
  awk '/MemFree|FilePages|SReclaimable/{s+=$4} END{print s+0}' \
    "/sys/devices/system/node/node${1}/meminfo" 2>/dev/null || echo 0
}

echo "=== Cocoon Hardware Profiling: $MODEL | batch=$LOGICAL_BATCH | GPUs=$GPUS ($NUM_GPUS) ==="

# ── Phase 1: Hardware snapshot ─────────────────────────────────────────────────
echo ""
echo "[1/4] Hardware snapshot"
if command -v nvidia-smi &>/dev/null; then
  GPU_FREE_MIB=$(CUDA_VISIBLE_DEVICES="$GPUS" \
    nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1 | tr -d ' ')
  GPU_TOTAL_MIB=$(CUDA_VISIBLE_DEVICES="$GPUS" \
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  GPU: total=${GPU_TOTAL_MIB} MiB  free=${GPU_FREE_MIB} MiB"
else
  echo "  WARNING: nvidia-smi not found."; GPU_FREE_MIB=0; GPU_TOTAL_MIB=0
fi

declare -A NUMA_TOTAL_KB
NUMA_NODES=()
for node_dir in /sys/devices/system/node/node*/; do
  [[ -f "${node_dir}meminfo" ]] || continue
  node=$(basename "$node_dir" | sed 's/node//')
  NUMA_NODES+=("$node")
  kb=$(numa_avail_kb "$node")
  total_kb=$(awk '/MemTotal/{print $4}' "${node_dir}meminfo" 2>/dev/null || echo 0)
  NUMA_TOTAL_KB[$node]=$total_kb
  printf "  NUMA node%s: %.2f GB available  %.2f GB total\n" \
    "$node" "$(echo "scale=2; $kb/1024/1024" | bc)" "$(echo "scale=2; $total_kb/1024/1024" | bc)"
done

# ── Phase 2: Physical batch size search ───────────────────────────────────────
# Run max_steps=1 DP-SGD probes (bandwidth=1) and halve on OOM.
echo ""
echo "[2/4] Physical batch size search (start=$(( LOGICAL_BATCH / NUM_GPUS )), halve on OOM)"
PHYSICAL_BS=$(( LOGICAL_BATCH / NUM_GPUS ))
FOUND_BS=0

for attempt in $(seq 1 8); do
  printf "  attempt %d: physical_bs=%d ... " "$attempt" "$PHYSICAL_BS"
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" $NUMACTL_PREFIX bash "$TRAIN_SCRIPT" \
    "$DATA_DIR" "$OUTPUT_DIR" "$TASK" "$MODEL" \
    "$PHYSICAL_BS" "$LOGICAL_BATCH" \
    "no" "1" "$NUM_GPUS" "$(( PORT + attempt ))" \
    "1" "1" "0" "MixGhostClip" "$TARGET_EPSILON" "automatic" "all-layer" \
    "no" "no" "no" "no" "1" "True" \
    > "$TMPDIR_PROF/probe_${attempt}.log" 2>&1
  set -e

  if grep -qiE "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" \
       "$TMPDIR_PROF/probe_${attempt}.log"; then
    echo "OOM"
    PHYSICAL_BS=$(( PHYSICAL_BS / 2 ))
    [[ $PHYSICAL_BS -lt 1 ]] && { echo "Error: batch size < 1. Model too large."; exit 1; }
  else
    FOUND_BS=$PHYSICAL_BS; echo "OK"; break
  fi
done
[[ $FOUND_BS -eq 0 ]] && { echo "Error: no working batch size found after 8 halvings."; exit 1; }

# ── Phase 3: Profiling run + background CPU sampler ───────────────────────────
# Sampler starts BEFORE training to capture model-load spike (can exceed steady-state).
echo ""
echo "[3/4] Profiling run (max_steps=3, GPU_partition=1, CPU_partition=0)"

declare -A NUMA_BASELINE_KB
for node in "${NUMA_NODES[@]}"; do
  NUMA_BASELINE_KB[$node]=$(numa_avail_kb "$node")
done

(
  while true; do
    parts=""
    for node in "${NUMA_NODES[@]}"; do parts="$parts ${node}:$(numa_avail_kb "$node")"; done
    echo "$parts" >> "$NUMA_SAMPLE_FILE"
    sleep 0.5
  done
) &
SAMPLER_PID=$!

CPU_VM_BASELINE_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().used/1e9:.3f}')")

set +e
CUDA_VISIBLE_DEVICES="$GPUS" $NUMACTL_PREFIX bash "$TRAIN_SCRIPT" \
  "$DATA_DIR" "$OUTPUT_DIR" "$TASK" "$MODEL" \
  "$FOUND_BS" "$LOGICAL_BATCH" \
  "no" "1" "$NUM_GPUS" "$(( PORT + 20 ))" \
  "1" "1" "0" "MixGhostClip" "$TARGET_EPSILON" "automatic" "all-layer" \
  "no" "no" "no" "no" "$PROFILE_STEPS" "True" \
  2>&1 | tee "$TRAIN_LOG"
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
unset SAMPLER_PID

# ── Phase 4: Parse + report ────────────────────────────────────────────────────
echo ""
echo "[4/4] Results"

# GPU peak: max torch.cuda.max_memory_allocated across all ranks (MiB, truncated to int)
GPU_PEAK_MIB=$(grep -oP 'torch\.cuda\.max_memory_allocated:\s*\K[0-9.]+' "$TRAIN_LOG" 2>/dev/null \
  | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{printf "%d\n", m}')
[[ -z "$GPU_PEAK_MIB" || "$GPU_PEAK_MIB" -eq 0 ]] && {
  echo "  WARNING: GPU peak not found in log; setting to 0."; GPU_PEAK_MIB=0; }

STEP_LATENCY_MS=$(grep -oP 'step_by_step\s*:\s*\K[0-9]+\.[0-9]+' "$TRAIN_LOG" 2>/dev/null | tail -1)
[[ -z "$STEP_LATENCY_MS" ]] && { echo "  WARNING: step latency not found."; STEP_LATENCY_MS=0; }

PARAMS=$(grep -oP 'Number of trainable params:\s*\K[0-9]+' "$TRAIN_LOG" 2>/dev/null | tail -1)
[[ -z "$PARAMS" ]] && { echo "  WARNING: trainable params not found."; PARAMS=0; }

declare -A NUMA_MIN_KB NUMA_DROP_KB
for node in "${NUMA_NODES[@]}"; do
  min_kb=$(awk -v n="$node" '
    { for (i=1;i<=NF;i++) { split($i,a,":"); if(a[1]==n && (min==""||a[2]+0<min+0)) min=a[2]+0 } }
    END { print (min==""?0:min) }
  ' "$NUMA_SAMPLE_FILE")
  NUMA_MIN_KB[$node]=$min_kb
  NUMA_DROP_KB[$node]=$(( NUMA_BASELINE_KB[$node] - min_kb ))
done

if [[ "$NUMA_NODE" == "auto" ]]; then
  ACTIVE_NODE=""; MAX_DROP=0
  for node in "${NUMA_NODES[@]}"; do
    (( ${NUMA_DROP_KB[$node]} > MAX_DROP )) && { MAX_DROP=${NUMA_DROP_KB[$node]}; ACTIVE_NODE=$node; } || true
  done
  [[ -z "$ACTIVE_NODE" ]] && ACTIVE_NODE="${NUMA_NODES[0]:-0}"
else
  ACTIVE_NODE=$NUMA_NODE
fi

CPU_VM_PEAK_GB=$(grep -oP 'CPU Virtual Memory:\s*used\s*=\s*\K[0-9.]+' "$TRAIN_LOG" 2>/dev/null \
  | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m+0}')
[[ -z "$CPU_VM_PEAK_GB" ]] && CPU_VM_PEAK_GB=0
NUMA_TOTAL_GB=$(echo "scale=3; ${NUMA_TOTAL_KB[$ACTIVE_NODE]:-0}/1024/1024" | bc)
CPU_VM_DELTA_GB=$(python3 -c "print(f'{max(0.0, ${CPU_VM_PEAK_GB} - ${CPU_VM_BASELINE_GB}):.3f}')")
CPU_FREE_GB=$(python3 -c "print(f'{max(0.0, ${NUMA_TOTAL_GB} - ${CPU_VM_DELTA_GB}):.2f}')")
CPU_FREE_NUMA_GB=$(python3 -c "print(f'{max(0.0, ${NUMA_MIN_KB[$ACTIVE_NODE]:-0}/1024/1024):.2f}')")
GPU_FREE_GB=$(echo "scale=2; ($GPU_TOTAL_MIB - $GPU_PEAK_MIB)/1024" | bc)
GPU_PEAK_GB=$(echo "scale=2; $GPU_PEAK_MIB/1024" | bc)
GPU_TOTAL_GB=$(echo "scale=2; $GPU_TOTAL_MIB/1024" | bc)

echo ""
echo "======================================================================"
printf " %-24s %s\n" "Model:"             "$MODEL"
printf " %-24s %d (largest fitting on GPU)\n" "Physical batch/GPU:" "$FOUND_BS"
printf " %-24s node%s\n" "Active NUMA node:"  "$ACTIVE_NODE"
printf " %-24s %s / %s GB (peak / total)\n" "GPU memory:"  "$GPU_PEAK_GB" "$GPU_TOTAL_GB"
printf " %-24s %s GB free\n"              "GPU free:"         "$GPU_FREE_GB"
printf " %-24s %s GB (delta method: NUMA_total - vm_delta)\n" "CPU free:" "$CPU_FREE_GB"
printf " %-24s baseline=%.3f  peak=%.3f  delta=%.3f GB\n" "  CPU VM:" "$CPU_VM_BASELINE_GB" "$CPU_VM_PEAK_GB" "$CPU_VM_DELTA_GB"
printf " %-24s %s GB (NUMA min-avail method, for comparison)\n" "  CPU free (NUMA):" "$CPU_FREE_NUMA_GB"
printf " %-24s %s ms\n"                   "Step latency:"     "$STEP_LATENCY_MS"
printf " %-24s %s\n"                     "Trainable params:"  "$PARAMS"
echo "======================================================================"
echo ""
echo "Pass these to select_partition.sh:"
echo "  GPU_FREE_GB=${GPU_FREE_GB} CPU_FREE_GB=${CPU_FREE_GB} STEP_LATENCY_MS=${STEP_LATENCY_MS} PHYSICAL_BS=${FOUND_BS} NUM_GPUS=${NUM_GPUS} PARAMS=${PARAMS}"
