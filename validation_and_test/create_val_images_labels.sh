#!/bin/bash
# Created Time: Wed 17 Feb 2021 05:09:22 PM PST
# Author: Taihong Xiao <xiaotaihong@126.com>

# create for pfpascal
# python create_val_images_labels.py \
#     --dataset pfpascal \
#     --split val \
#     --arch resnet50 \
#     --modelpath ./ckpt/checkpoint_0062.pth.tar \
#     --downscale 16 \
#     --patch_size 128 \
#     --rotate 10 \
#     --moco_dim 128 \
#     --temp 0.0007 \
#     --gpu 0 \
#     --save_dir ./crop/

# # create for pfwillow
# python create_val_images_labels.py \
#     --dataset pfwillow \
#     --split val \
#     --arch resnet50 \
#     --modelpath ./ckpt/checkpoint_0062.pth.tar \
#     --downscale 16 \
#     --patch_size 128 \
#     --rotate 10 \
#     --moco_dim 128 \
#     --temp 0.0007 \
#     --gpu 0 \
#     --save_dir ./crop/

# create for spair
python create_val_images_labels.py \
    --dataset spair \
    --split val \
    --arch resnet50 \
    --modelpath ./ckpt/checkpoint_0062.pth.tar \
    --downscale 16 \
    --patch_size 128 \
    --rotate 10 \
    --moco_dim 128 \
    --temp 0.0007 \
    --gpu 4 \
    --save_dir ./crop/

