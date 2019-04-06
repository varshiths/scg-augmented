#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -b 32 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nob/vgrel-16.tar \
    -cache cache/sgdet_motifnet-size-predcls-nob_vgrel-16
    # -use_bias

# echo "***************************"
# echo "***************************"

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 32 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-vg-1.0-0.5/vgrel-99.tar \
#     -cache cache/sgdet_motifnet-size-predcls-vg-1.0-0.5_vgrel-99

# echo "***************************"
# echo "***************************"

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 32 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-coco-1.0-0.5/vgrel-99.tar \
#     -cache cache/sgdet_motifnet-size-predcls-coco-1.0-0.5_vgrel-99

# echo "***************************"
# echo "***************************"

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -order size \
#     -b 32 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-hid-1.0-0.5/vgrel-99.tar \
#     -cache cache/sgdet_motifnet-size-predcls-hid-1.0-0.5_vgrel-99

# echo "***************************"
# echo "***************************"
