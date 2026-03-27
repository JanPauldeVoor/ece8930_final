#!/bin/bash

export HF_HUB_OFFLINE=1

lerobot-train \
    --dataset.repo_id=local_datasets/so101_touch_cube \
    --dataset.root=./local_datasets/so101_touch_cube \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --wandb.enable=false \
    --policy.compile_model=true \
    --policy.repo_id=so101_touch_cube_ \
    --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=1000 \
    --policy.device=cuda \
    --batch_size=2