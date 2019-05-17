#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# Number of images in train set
# 108073
python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -no_bg \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/motifnet-size-predcls-nobg-nob
    # -num_im 30000
    
python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -bias_src vg \
    -prior_weight 1.0 \
    -distillation_weight 0.5 \
    -no_bg \
    -save_dir checkpoints/motifnet-size-predcls-nobg-vg-1.0-0.5
    # -num_im 30000

python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -bias_src coco \
    -prior_weight 1.0 \
    -distillation_weight 0.5 \
    -no_bg \
    -save_dir checkpoints/motifnet-size-predcls-nobg-coco-1.0-0.5
    # -num_im 30000

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
    -no_bg \
    -save_dir checkpoints/motifnet-size-predcls-nobg-hid-1.0-0.5
    # -num_im 30000
