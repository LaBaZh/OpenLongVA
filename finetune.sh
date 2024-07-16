#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export BATCH_SIZE=4
export GRADIENT_ACCU_STEPS=1
export MASTER_PORT=29502
export CPUS_PER_TASK=24
export QUOTA=reserved

export DATA_PATH=data/subset.json
export SAVE_PATH=test_train
export BASE_LR=2e-5
export VIT_LR=2e-6


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port ${MASTER_PORT} \
    longva/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path pretrained/LongVA-7B \
    --version qwen_1_5 \
    --data_path ${DATA_PATH} \
    --image_folder data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower False \
    --mm_vision_tower_lr ${VIT_LR} \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type unires \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SAVE_PATH}
