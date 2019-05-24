#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# python models/train_rels_it.py \
#    -m predcls \
#    -model motifnet \
#    -order size \
#    -b 2 \
#    -p 100 \
#    -ngpu 1 \
#    -ckpt checkpoints/vgdet/vg-24.tar \
#    -teacher1_ckpt checkpoints/motifnet-size-predcls-nob/vgrel-16.tar \
#    -bias_src hid \
#    -prior_weight 1.0 \
#    -no_bg \
#    -distillation_weight 0.5 \
#    -save_dir checkpoints/temp
#    # -num_im 30000

# python models/train_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 8 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/vgdet/vg-24.tar \
#     -bias_src coco \
#     -prior_weight 1.0 \
#     -distillation_weight 0.5 \
#     -no_bg \
#     -save_dir checkpoints/temp

python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 1000 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -bias_src sample.npy  \
    -prior_weight 1.0 \
    -distillation_weight 0.5 \
    -save_dir checkpoints/temp
    # -no_bg \
