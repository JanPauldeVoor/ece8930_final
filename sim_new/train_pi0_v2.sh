#!/bin/bash

#export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-train \
    --dataset.repo_id=local_datasets/so101_touch_cube \
    --dataset.root=./local_datasets/so101_touch_cube \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --wandb.enable=false \
    --policy.compile_model=false \
    --policy.repo_id=so101_touch_cube_ \
    --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=1
