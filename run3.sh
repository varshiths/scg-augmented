#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# python models/train_rels.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -save_dir checkpoints/motifnet-size-predcls-nob
#     # -use_bias

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -bias_src vg \
#     -save_dir checkpoints/motifnet-size-predcls-vg-1.0-0.5 \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/motifnet-size-predcls-nob/vgrel-5.tar \
#     -bias_src coco \
#     -save_dir checkpoints/motifnet-size-predcls-coco-0.8-0.3 \
#     -prior_weight 0.8 \
#     -distillation_weight 0.3

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/motifnet-size-predcls-nob/vgrel-5.tar \
#     -bias_src coco \
#     -save_dir checkpoints/motifnet-size-predcls-coco-1.0-0.5 \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5

python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/motifnet-size-predcls-vg-1.0-0.5/vgrel-10.tar \
    -bias_src vg \
    -save_dir checkpoints/motifnet-size-predcls-coco-exp \
    -prior_weight 1.0 \
    -distillation_weight 0.3

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -bias_src hid \
#     -save_dir checkpoints/motifnet-size-predcls-hid-1.0-0.5 \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5
