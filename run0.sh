#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

python models/train_rels_it.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 2 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -teacher1_ckpt checkpoints/motifnet-size-predcls-nob/vgrel-16.tar \
    -bias_src hid \
    -prior_weight 1.0 \
    -nbg \
    -distillation_weight 0.5 \
    -save_dir checkpoints/temp
    # -num_im 30000
