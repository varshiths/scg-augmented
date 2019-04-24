#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-nbg-nob/vgrel-5.tar \
#     -cache cache/predcls_motifnet-size-predcls-nbg-nob_vgrel-5

# python models/_visualize.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-nbg-nob/vgrel-5.tar \
#     -cache cache/predcls_motifnet-size-predcls-nbg-nob_vgrel-5

# python models/_visualize.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-nbg-vg-1.0-0.5/vgrel-5.tar \
#     -cache cache/predcls_motifnet-size-predcls-nbg-vg-1.0-0.5_vgrel-5

python models/_visualize.py \
    -m predcls \
    -model motifnet \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-coco-1.0-0.5/vgrel-6.tar \
    -cache cache/predcls_motifnet-size-predcls-nbg-coco-1.0-0.5_vgrel-6

mv qualitative/ results/qualitative-nbg-coco
mkdir qualitative

python models/_visualize.py \
    -m predcls \
    -model motifnet \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-hid-1.0-0.5/vgrel-9.tar \
    -cache cache/predcls_motifnet-size-predcls-nbg-hid-1.0-0.5_vgrel-9

mv qualitative/ results/qualitative-nbg-hid
mkdir qualitative
