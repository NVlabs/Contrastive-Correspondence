#!/bin/bash
# Created Time: Wed 02 Sep 2020 12:51:26 PM PDT
# Author: Taihong Xiao <xiaotaihong@126.com>

# cd /raid;
# tar xf /imagenet2012_tar/imagenet2012.tar.gz;
# cd /workspace/moco_cycle_fix;

python -W ignore:semaphore_tracker:UserWarning main_moco_icycle_pf_aug_attention_relu.py \
    -a resnet50 \
    --norm true \
    --dist-url 'tcp://localhost:10004' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dataset spair \
    --image_dir ../ImageNet2012/train \
    --datapath ../Datasets_SCOT \
    --restore ../pretrained_models/moco.pth.tar \
    --savedir ../train_log/moco_icycle_spair_aug_attention_relu-gpu_8-lr_0.003-bs_256_layer_13-downscale_16-drop_0-icycle_lw_0.0005-temp_0.0007-lc_0 \
    --device 8 \
    --lr 0.003 \
    --lr_mode step\
    --lr_schedule 5 10 15 \
    --batch_size 256 \
    --patch_size 128 \
    --frame_num 2 \
    --topk 1 \
    --uselayer 13 \
    --downscale 16 \
    --temp 0.0007 \
    --dropout_rate 0 \
    --nepoch 20 \
    --log_interval 5000 \
    --save_interval 5000 \
    --icycle_lw 0.0005 \
    --lc 0

