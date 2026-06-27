#!/bin/bash

# DynEval curriculum fine-tuning for Qwen3-VL-4B-Instruct.
# Usage:
#   export DYNEVALINSTRUCT_T2IA_ANNOTATION=/path/to/dynevalinstruct_t2ia.json
#   export DYNEVALINSTRUCT_T2IA_DATA=/path/to/images
#   bash scripts/sft_qwen_3_vl_4b.sh 1
#
#   export DYNEVALINSTRUCT_IQA_ANNOTATION=/path/to/dynevalinstruct_iqa.json
#   export DYNEVALINSTRUCT_IQA_DATA=/path/to/images
#   bash scripts/sft_qwen_3_vl_4b.sh 2

set -euo pipefail

STAGE=${1:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}
if [ -z "${NPROC_PER_NODE}" ] || [ "${NPROC_PER_NODE}" -lt 1 ]; then
    NPROC_PER_NODE=1
fi

deepspeed=./scripts/zero3.json
llm=Qwen/Qwen3-VL-4B-Instruct
additional_special_tokens="<|T2IA|>,<|IQA|>,<|EVALUATION|>"
lr=1e-5
batch_size=4
grad_accum_steps=4
entry_file=qwenvl/train/train_qwen.py
report_to=${REPORT_TO:-none}

if [ "${STAGE}" -eq 1 ]; then
    : "${DYNEVALINSTRUCT_T2IA_ANNOTATION:?Set DYNEVALINSTRUCT_T2IA_ANNOTATION to the Stage 1 JSON file}"
    : "${DYNEVALINSTRUCT_T2IA_DATA:?Set DYNEVALINSTRUCT_T2IA_DATA to the Stage 1 image root}"
    datasets=dynevalinstruct_t2ia
    run_name="dyneval_4b_stage1_t2ia"
    output_dir=./output/dyneval_4b_stage1_t2ia
elif [ "${STAGE}" -eq 2 ]; then
    : "${DYNEVALINSTRUCT_IQA_ANNOTATION:?Set DYNEVALINSTRUCT_IQA_ANNOTATION to the Stage 2 JSON file}"
    : "${DYNEVALINSTRUCT_IQA_DATA:?Set DYNEVALINSTRUCT_IQA_DATA to the Stage 2 image root}"
    datasets=dynevalinstruct_iqa
    run_name="dyneval_4b_stage2_iqa"
    output_dir=./output/dyneval_4b_stage2_iqa
    llm=./output/dyneval_4b_stage1_t2ia
    if [ ! -f "${llm}/config.json" ]; then
        echo "Stage 2 requires the Stage 1 checkpoint at ${llm}"
        echo "Run: bash scripts/sft_qwen_3_vl_4b.sh 1"
        exit 1
    fi
else
    echo "Unknown stage: ${STAGE}. Use 1 for T2IA or 2 for IQA."
    exit 1
fi

echo "Stage ${STAGE}: model=${llm}, dataset=${datasets}, output=${output_dir}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${entry_file}" \
    --deepspeed "${deepspeed}" \
    --model_name_or_path "${llm}" \
    --additional_special_tokens "${additional_special_tokens}" \
    --dataset_use "${datasets}" \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir "${output_dir}" \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size "$((batch_size * 2))" \
    --gradient_accumulation_steps "${grad_accum_steps}" \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate "${lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "${run_name}" \
    --report_to "${report_to}"
