#!/usr/bin/env bash

# for testing on benchmark dataset
# dataset_name: GRR_real  RSGR-GS_v1
#Substitute the "model_dir" and "input_dir" with your pretrained model and data path


CUDA_VISIBLE_DEVICES=2 python3 test.py \
        --win_size_h=4 --win_size_w=16 --trans_type='rsst' --embed_dim=32 \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='GRR_real' \
		--output_dir='./output' --keep_frames \
		--model_dir='./train_log/GRR_real/checkpoint.ckpt' --batch_size=1