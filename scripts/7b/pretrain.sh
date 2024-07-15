#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=2
export NNODES=1
export BATCH_SIZE=1
export GRADIENT_ACCU_STEPS=1
export MASTER_PORT=29502
export CPUS_PER_TASK=24
export QUOTA=reserved

export DATA_PATH=/data3/Open-LLaVA-NeXT/data/llava/llava-pretrain/blip_laion_cc_sbu_558k.json
export SAVE_PATH=llava-v1.6-7b_qwen-7b_pretrain_lcs-558k_ft-mlp-lr-1e-3
export BASE_LR=1e-3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --master_addr localhost --master_port ${MASTER_PORT} \
longva/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path pretrained/Qwen2-7B-Instruct-224K \
--version qwen_1_5 \
--data_path ${DATA_PATH} \
--image_folder /data3/Open-LLaVA-NeXT/data/llava/llava-pretrain/images \
--vision_tower /data3/Open-LLaVA-NeXT/openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--tune_mm_mlp_adapter True \
--unfreeze_mm_vision_tower False \
--image_aspect_ratio anyres \
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
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to wandb \
--run_name ${SAVE_PATH}