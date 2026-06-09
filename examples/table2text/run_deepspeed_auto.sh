#!/usr/bin/env bash
# Manual tester for different partition setups (PARTITION, GPU_PART, CPU_PART).
# Sweeps over a user-specified list of band sizes and batch sizes and finds the
# largest viable batch per band.  Use this to evaluate a specific partition config
# before committing to a full run.  For automated partition selection and end-to-end
# timing, use run_llm.sh instead.
#
# Example usage:
#   PARTITION=1 GPU_PART=1 CPU_PART=0 MODEL=opt13 BASE_MAX_STEPS=1024 \
#     BANDS="8 16 32" BATCH_SIZES="128 64 32 16" bash run_deepspeed_auto.sh

#   PARTITION=20 GPU_PART=1 CPU_PART=5 MODEL=opt13 BASE_MAX_STEPS=1024 \
#     BANDS="64" BATCH_SIZES="32" bash run_deepspeed_auto.sh

# --- 1. PARAMETER INPUTS ---
PARTITION=${PARTITION:?"Error: PARTITION must be set."}
GPU_PART=${GPU_PART:?"Error: GPU_PART must be set."}
CPU_PART=${CPU_PART:?"Error: CPU_PART must be set."}
MODEL=${MODEL:?"Error: MODEL must be set."}
BASE_MAX_STEPS=${BASE_MAX_STEPS:?"Error: BASE_MAX_STEPS must be set."}

# Lists that contain the hyperparameter search space (space-separated)
BANDS=${BANDS:-"32"}               # Default to "32"
BATCH_SIZES=${BATCH_SIZES:-"32"}   # Default to "32" (Largest first!)

# --- 2. CONSTANT CONFIGURATION ---
GPU="4,5,6,7"
NUM_GPU="4"
NUMA_CPUS="${NUMA_CPUS:-28-55,84-111}"
NUMA_MEM="${NUMA_MEM:-1}"
TRAIN_SCRIPT="table2text/run_ZERO1.sh"
PREFIX_PATH="${PREFIX_PATH:-./data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output}"
DATASET="e2e"
DTYPE="BF16"
PORT=9800

# --- 3. VALIDATION AND INITIALIZATION ---
# Check if the partition size is valid (simple sum check)
if [[ $(( GPU_PART + CPU_PART )) -gt $PARTITION ]]; then
    echo "ERROR: GPU_PART (${GPU_PART}) + CPU_PART (${CPU_PART}) is greater than PARTITION (${PARTITION}). Exiting."
    exit 1
fi

# BIGGEST_BATCH is initialized to the BASE_MAX_STEPS, as requested.
BIGGEST_BATCH=${BASE_MAX_STEPS}

echo "=========================================================================="
echo "Starting Single-Run Optimizer"
echo "Configuration: MODEL=${MODEL} | P=${PARTITION} | G=${GPU_PART} | C=${CPU_PART} | STEPS=${BASE_MAX_STEPS}"
echo "BANDS to test: ${BANDS}"
echo "Batch Sizes (Largest first): ${BATCH_SIZES}"
echo "=========================================================================="


# --- 4. CORE LOGIC: ITERATE OVER BANDS AND BATCH SIZES ---

# Loop over the parameter that drives the core optimization (bands)
for band in $BANDS; do
    
    echo -e "\n--- Testing BAND: ${band} ---"
    
    BEST_BATCH=0
    
    # Loop over the optimization parameter (batch sizes)
    # Note: BATCH_SIZES list must be sorted largest-to-smallest for this logic to work!
    for bsz in $BATCH_SIZES; do
        
        # Check against the largest successful batch size found so far (initialized to BASE_MAX_STEPS)
        if [[ "${bsz}" -gt "${BIGGEST_BATCH}" ]]; then
            continue
        fi
        
        echo -e "  Trying batch size ${bsz}..."
        
        # --- CALCULATIONS ---
        LOGFILE="logg_${MODEL}_S${BASE_MAX_STEPS}_P${PARTITION}G${GPU_PART}C${CPU_PART}_BAND${band}_BSZ${bsz}_test4.txt"
        
        multiple=$(( NUM_GPU * bsz ))
        # Calculate MAX_STEPS using the fixed BASE_MAX_STEPS
        MAX_STEPS=$(( (BASE_MAX_STEPS / multiple) * multiple ))

        # --- EXECUTION ---
        CUDA_VISIBLE_DEVICES="$GPU" \
        numactl -C "$NUMA_CPUS" -m "$NUMA_MEM" \
        bash "$TRAIN_SCRIPT" \
             "$PREFIX_PATH" \
             "$OUTPUT_PATH" \
             "$DATASET" \
             "$MODEL" \
             "$bsz" \
             "$MAX_STEPS" \
             "no" \
             "$band" \
             "$NUM_GPU" "$PORT" \
             "$PARTITION" "$GPU_PART" "$CPU_PART" \
             2>&1 | tee "$LOGFILE"

        # --- OOM CHECK ---
        if grep -qi "out of memory" "$LOGFILE"; then
            echo "[OOM detected] Reducing batch size and retry..."
        else
            echo "[SUCCESS] Batch size $bsz works."
            BIGGEST_BATCH=$bsz # Update the largest successful batch size
            BEST_BATCH=$bsz
            # Break the BATCH_SIZES loop because we found the largest feasible batch for this band
            break
        fi
    done # End of BATCH_SIZES loop

    # --- SUMMARY FOR THE CURRENT BAND ---
    if [ "$BEST_BATCH" -eq 0 ]; then
        echo "--- FAILURE: No viable batch size found for BAND=${band}. ---"
    else
        echo "--- SUCCESS: BEST BATCH SIZE for BAND=${band} = $BEST_BATCH ---"
    fi

done # End of BANDS loop

echo -e "\n--------------------------------------------"
echo "Optimization run completed for all bands."
echo "--------------------------------------------"