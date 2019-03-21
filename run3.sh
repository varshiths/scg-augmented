#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# # already trained
# python models/train_rels_tc.py \
#     -m sgdet \
#     -model motifnet \
#     -b 8 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -save_dir checkpoints/motifnet-leftright-sgdet-vg \
#     -bias_src vg \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5

python models/train_rels_tc.py \
    -m sgdet \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/motifnet-size-sgdet-rc-1.0-0.5 \
    -bias_src rc \
    -prior_weight 1.0 \
    -distillation_weight 0.5

python models/train_rels_tc.py \
    -m sgdet \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/motifnet-size-sgdet-vg-1.0-0.5 \
    -bias_src vg \
    -prior_weight 1.0 \
    -distillation_weight 0.5

# python models/train_rels_tc.py \
#     -m sgdet \
#     -model motifnet \
#     -b 8 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -save_dir checkpoints/motifnet-leftright-sgdet-vg \
#     -bias_src vg \
#     -prior_weight 1.0 \
#     -distillation_weight 0.6
