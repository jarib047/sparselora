#!/usr/bin/env bash
# set -e

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL_PATH=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
MODEL_PATH=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
SPARSELORA_PATH=${2:-"z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA"}
SEED=42

PYTHONPATH=/home/bsilwal torchrun --nproc_per_node=gpu experiments/train.py \
    --model_name_or_path $MODEL_PATH \
    --dataset datasets/csr170k.json \
    --sparselora path=$SPARSELORA_PATH,mode=o1,start_step=0.05 \
    --output_dir checkpoints/csr170k_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.04 \
    --seed $SEED \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --ddp_find_unused_parameters false \
    --peft lora

PYTHONPATH=/home/bsilwal torchrun --nproc_per_node=gpu experiments/eval.py \
    --model_name_or_path checkpoints/csr170k_lora \
    --dataset boolq+piqa+social_i_qa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa


# !/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OMP_NUM_THREADS=8
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL_PATH=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}
# SPARSELORA_PATH=${2:-"z-lab/Meta-Llama-3-8B-Instruct-SparseLoRA"}
# SEED=42

# torchrun --nproc_per_node=4 experiments/train.py \
#     --model_name_or_path "$MODEL_PATH" \
#     --dataset datasets/csr170k.json \
#     --sparselora path="$SPARSELORA_PATH",mode=o1,start_step=0.05 \
#     --output_dir checkpoints/csr170k \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 3e-4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.04 \
#     --seed "$SEED" \
#     --bf16 true \
#     --logging_steps 1 \
#     --save_strategy no \
#     --report_to none \
#     --ddp_find_unused_parameters false

# torchrun --nproc_per_node=4 experiments/eval.py \
#     --model_name_or_path checkpoints/csr170k \
#     --dataset boolq+piqa+social_i_qa+hellaswag+winogrande+arc-easy+arc-challenge+openbookqa
