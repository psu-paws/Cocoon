#!/usr/bin/env bash
# Getting-started checks — expected runtime ~15-30 minutes.
#
# Three checks:
#   [1] CPU/GPU GEMV throughput  (PyTorch install sanity + noise compute timing)
#   [2] Noise worker pipeline    (verify CPU GEMV is fully hidden behind training)
#   [3] Noise preprocessing      (DLRM cold-row accumulation correctness)
#
# Override defaults from the environment before running, e.g.:
#   NUM_GPUS=8 THREADS=28 BAND=64 bash run_getting_started.sh
#
# Target hardware: 4 × NVIDIA A5000 24 GB, 128 GB RAM, PCIe 4.0, 56-core Xeon.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# ── Configurable ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"
THREADS="${THREADS:-14}"            # OMP threads per CPU worker (use ~quarter of cores)
# CPU cores for noise workers.  Single-worker checks use NUMA_CPUS directly.
# Multi-GPU check [2b] uses NUMA_CPUS_MULTI (one THREADS-wide slice per GPU).
# Format: comma-separated ranges, e.g. "21-27,77-83" (= 14 cores for 1 worker).
NUMA_CPUS="${NUMA_CPUS:-21-27,77-83}"
NUMA_CPUS_MULTI="${NUMA_CPUS_MULTI:-21-27,77-83:28-34,84-90:35-41,91-97:42-48,98-104}"
BAND="${BAND:-32}"                  # BandMF bandwidth for GEMV throughput check
NOISE_PARTITION="${NOISE_PARTITION:-4}"  # total P = PART_GPU(1) + PART_CPU(5)
MODEL_PARAMS="${MODEL_PARAMS:-356929536}"  # opt-350M
STEP_MS="${STEP_MS:-2000}"           # expected training step latency (ms)
STEPS="${STEPS:-30}"                # number of steps for pipeline check
PART_CPU="${PART_CPU:-3}"
PART_CXL=$(( NOISE_PARTITION - 1 - (PART_CPU) ))
UNIT_SIZE=$(( MODEL_PARAMS / (NUM_GPUS * NOISE_PARTITION) ))

# numactl wrapper: pins process (and all subprocesses) to given CPUs before Python starts,
# so OpenBLAS initializes within the restricted affinity rather than seeing all cores.
numactl_pin() {
  local cpus="$1"; shift
  if command -v numactl &>/dev/null; then
    numactl --physcpubind="$cpus" "$@"
  else
    "$@"
  fi
}

BM="$REPO_ROOT/benchmark"

echo "========================================================================"
echo " Cocoon Getting-Started Checks"
echo "========================================================================"
printf "  model_params     = %s\n" "$MODEL_PARAMS"
printf "  n_gpus           = %s\n" "$NUM_GPUS"
printf "  noise_partition  = %s  (PART_GPU=1, PART_CPU=%s)\n" "$NOISE_PARTITION" "$PART_CPU"
printf "  unit_size        = %s  params/worker partition\n" "$UNIT_SIZE"
printf "  band             = %s\n" "$BAND"
printf "  omp_threads      = %s\n" "$THREADS"
printf "  simulated_step   = %s ms\n" "$STEP_MS"
echo "========================================================================"
echo ""

# ── [1] BandMF GEMV correctness + throughput ────────────────────────────────
echo "=== Check [1/4]: BandMF GEMV correctness ==="
echo ""
echo "    Verifies step() output matches explicit recurrence on a 5-step example."
echo "    Verifies CPU device and CPU→GPU transfer are correct."
echo ""
$PYTHON "$BM/verify_bandmf_step.py"
echo ""

echo "=== Check [2/4]: CPU/GPU GEMV throughput ==="
echo ""

echo "--- [2a] CPU GEMV, single partition ---"
echo "    num_params=${UNIT_SIZE}  threads=${THREADS}  band=${BAND}  cpus=${NUMA_CPUS}"
numactl_pin "$NUMA_CPUS" $PYTHON "$BM/bench_gemv_cpu.py" "$UNIT_SIZE" "$THREADS" "$BAND"
echo ""

if [[ "$NUM_GPUS" -gt 1 ]]; then
    TOTAL_PER_GPU=$(( UNIT_SIZE * NUM_GPUS ))
    echo "--- [2b] CPU GEMV, ${NUM_GPUS}-GPU (one worker per GPU) ---"
    echo "    num_params=${TOTAL_PER_GPU}  threads=${THREADS}  band=${BAND}  n_gpus=${NUM_GPUS}  cpus=${NUMA_CPUS_MULTI}"
    # Pass per-GPU CPU specs as 5th arg so each worker pins itself to its own slice.
    # The outer numactl restricts the parent; bench_gemv_cpu.py refines per-worker via sched_setaffinity.
    ALL_CPUS=$(echo "$NUMA_CPUS_MULTI" | tr ':' ',')
    numactl_pin "$ALL_CPUS" $PYTHON "$BM/bench_gemv_cpu.py" "$TOTAL_PER_GPU" "$THREADS" "$BAND" "$NUM_GPUS" "$NUMA_CPUS_MULTI"
    echo ""
else
    echo "--- [2b] skipped (NUM_GPUS=1, same as [2a]) ---"
    echo ""
fi

echo "--- [2c] GPU GEMV (on-device, no H2D) ---"
echo "    num_params=${UNIT_SIZE}  band=${BAND}"
$PYTHON "$BM/bench_gemv_gpu.py" "$UNIT_SIZE" "$BAND"
echo ""

# ── [2] Noise worker pipeline sanity check ───────────────────────────────────
echo "=== Check [3/4]: Noise worker pipeline ==="
echo ""
echo "    Simulates ${STEPS} training steps of ${STEP_MS} ms each."
echo "    Measures queue wait per step — should be near 0 ms if GEMV < step latency."
echo ""
numactl_pin "$NUMA_CPUS" $PYTHON "$BM/verify_noise_pipeline.py" \
    --num-params "$UNIT_SIZE" \
    --band-size  "$BAND" \
    --partition  "$PART_CPU" \
    --step-ms    "$STEP_MS" \
    --steps      "$STEPS" \
    --threads    "$THREADS"
echo ""

# ── [3] Noise preprocessing correctness ──────────────────────────────────────
echo "=== Check [4/4]: Noise preprocessing correctness (DLRM cold-row accumulation) ==="
echo ""
echo "    Compares NoiseCache coalesced path against per-row naive BandMF simulation."
echo "    Max absolute error should be < 1e-5. "
echo ""
$PYTHON "$BM/verify_noise_preprocessing.py"
echo ""

echo "========================================================================"
echo " Getting-started checks complete."
echo " If all 4 checks passed, proceed to the Detailed Instructions in README.md."
echo "========================================================================"
