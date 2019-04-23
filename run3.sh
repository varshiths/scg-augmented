#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -save_dir checkpoints/motifnet-size-predcls-nbg-nob

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -bias_src vg \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5 \
#     -save_dir checkpoints/motifnet-size-predcls-nbg-vg-1.0-0.5

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -bias_src coco \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5 \
#     -save_dir checkpoints/motifnet-size-predcls-nbg-coco-1.0-0.5

python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -bias_src hid \
    -prior_weight 1.0 \
    -distillation_weight 0.5 \
    -save_dir checkpoints/motifnet-size-predcls-nbg-hid-1.0-0.5
