#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

# evaluation script for hyper param search

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/motifnet-size-predcls-nob/vgrel-16.tar \
#     -cache cache/predcls_motifnet-size-predcls-nob_vgrel-16_eval
#     # -test \

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-coco-0.8-0.3/vgrel-4.tar \
#     -cache cache/predcls_motifnet-size-predcls-coco-0.8-0.3_vgrel-4

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-coco-1.0-0.5/vgrel-4.tar \
#     -cache cache/predcls_motifnet-size-predcls-coco-1.0-0.5_vgrel-4

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -ckpt checkpoints/motifnet-size-predcls-coco-0.8-0.5/vgrel-20.tar \
#     -cache cache/predcls_motifnet-size-predcls-coco-0.8-0.5_vgrel-20_eval
#     # -test \

# python models/eval_rels_tc.py \
#     -m predcls \
#     -model motifnet \
#     -b 16 \
#     -p 100 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-size-predcls-coco-1.0-0.5/vgrel-4.tar

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-coco-b-1.0-0.5/vgrel-8.tar

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-hid-b-1.0-0.5/vgrel-5.tar
