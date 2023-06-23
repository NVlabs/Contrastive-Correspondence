#!/bin/bash
# Created Time: Wed 17 Feb 2021 05:09:22 PM PST
# Author: Taihong Xiao <xiaotaihong@126.com>

python create_val_images_labels_2.py \
    --dataset pfpascal \
    --split val \
    --arch resnet50 \
    --modelpath ./ckpt/checkpoint_0062.pth.tar \
    --downscale 16 \
    --patch_size 128 \
    --rotate 10 \
    --moco_dim 128 \
    --temp 0.0007 \
    --save_dir ./val_images

