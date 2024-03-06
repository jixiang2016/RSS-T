#!/usr/bin/env bash

#dataset_name:GRR_real, RSGR-GS_v1
#When changing "window size", please update "padding factor" in "trainer.py" !!!!!
#Substitute the "input_dir" with your data path

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_port=17634 --nproc_per_node=4 train.py \
        --world_size=4 \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='GRR_real' \
		--output_dir='./train_log' --batch_size_val=2 \
		--batch_size=1 --epoch=600 --win_size_h=4 --win_size_w=16 --trans_type='rsst' --embed_dim=32


