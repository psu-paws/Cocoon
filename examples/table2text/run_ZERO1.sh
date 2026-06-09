#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INT64_MAX=9223372036854775807  # sentinel: no limit

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
data_dir=${1:-"${REPO_ROOT}/data/prefix-tuning"}
output_dir=${2:-"${REPO_ROOT}/output"}
task_mode=${3:-"e2e"}
model_name_or_path=${4:-"gpt2-medium"}  # one of: distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl gptj
physical_batch_size=${5:-40}
batch_size=${6:-1000}
non_private=${7:-"no"}
min_separation=${8:-1}
num_GPUs=${9:-8}
masterport=${10:-"61000"}
noise_partition=${11:-1}
GPU_partition=${12:-1}
CPU_partition=${13:-0}
clipping_mode=${14:-"MixGhostClip"}
target_epsilon=${15:-8}
clipping_fn=${16:-"automatic"}
clipping_style=${17:-"all-layer"}
bias_only=${18:-"no"}
attention_only=${19:-"no"}
static_lm_head=${20:-"no"}
static_embedding=${21:-"no"}
max_steps=${22:-0}   # 0 = use num_train_epochs; >0 = stop after N optimizer steps
speed_mode=${23:-"True"}  # True = skip BandMF factorization (random matrix, timing only)

export PATH="${CONDA_PREFIX:-$(dirname "$(which python)")}:$PATH"

DS_CONFIG="${SCRIPT_DIR}/.ds_config_$$.json"
trap "rm -f ${DS_CONFIG}" EXIT
cat > "${DS_CONFIG}" <<EOF
{
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "train_batch_size": ${batch_size},
  "train_micro_batch_size_per_gpu": ${physical_batch_size},
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true,
  "zero_optimization": { "stage": 1 }
}
EOF

if [[ ${task_mode} == "e2e" ]]; then
  data_dir="${data_dir}/data/e2e_data"
  target_delta=8e-6
  num_train_epochs=1
  max_seq_len=100
  if [[ ${bias_only} == "yes" ]]; then
    learning_rate=1e-2
  else
    learning_rate=1e-1
  fi
elif [[ ${task_mode} == "dart" ]]; then
  target_delta=1e-5
  data_dir="${data_dir}/data/dart"
  num_train_epochs=15  # approximately same number of updates as e2e
  max_seq_len=120
  if [[ ${bias_only} == "yes" ]]; then
    learning_rate=2e-3
  else
    learning_rate=5e-4  # lower lr for stability in large models
  fi
else
  echo "Unknown task: ${task_mode}"
  exit 1
fi

deepspeed \
  --master_port ${masterport} \
  "${SCRIPT_DIR}/run_language_modeling.py" \
  --deepspeed_config ${DS_CONFIG} \
  --output_dir ${output_dir} --overwrite_output_dir \
  --task_mode ${task_mode} \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --do_train --do_eval \
  --line_by_line \
  --save_steps 100000 --save_total_limit 1 --save_at_last no \
  --logging_dir ${output_dir} --logging_steps 1 \
  --seed 0 --min_separation ${min_separation} \
  --dataloader_num_workers 2 \
  --eval_steps -100000 --eval_epochs 999 --max_eval_batches 100 \
  --evaluation_strategy epoch --evaluate_before_training "no" --evaluate_during_training "no" \
  --per_device_eval_batch_size 10 \
  --max_generations ${INT64_MAX} --max_generations_train 10 --max_generations_valid ${INT64_MAX} \
  --max_train_examples ${INT64_MAX} --max_valid_examples ${INT64_MAX} --max_eval_examples ${INT64_MAX} \
  --data_folder ${data_dir} --max_seq_len ${max_seq_len} --format_mode cat \
  --per_example_max_grad_norm 0.1 --target_delta ${target_delta} --target_epsilon ${target_epsilon} \
  --learning_rate ${learning_rate} --lr_decay "no" \
  --num_train_epochs ${num_train_epochs} \
  --per_device_train_batch_size ${physical_batch_size} --logical_batch_size ${batch_size} \
  --attention_only ${attention_only} --bias_only ${bias_only} \
  --static_lm_head ${static_lm_head} --static_embedding ${static_embedding} \
  --non_private ${non_private} \
  --noise_partition ${noise_partition} --GPU_partition ${GPU_partition} --CPU_partition ${CPU_partition} \
  --num_GPUs ${num_GPUs} \
  --clipping_mode "${clipping_mode}" --clipping_fn "${clipping_fn}" --clipping_style "${clipping_style}" \
  --max_steps ${max_steps} \
  --speed_mode ${speed_mode}
