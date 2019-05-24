#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nobg-nob/vgrel-5.tar \
    -cache cache/predcls_motifnet-size-predcls-nobg-nob_vgrel-5

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nobg-vg-1.0-0.5/vgrel-4.tar \
    -cache cache/predcls_motifnet-size-predcls-nobg-vg-1.0-0.5_vgrel-4

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nobg-coco-1.0-0.5/vgrel-4.tar \
    -cache cache/predcls_motifnet-size-predcls-nobg-coco-1.0-0.5_vgrel-4

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nobg-hid-1.0-0.5/vgrel-7.tar \
    -cache cache/predcls_motifnet-size-predcls-nobg-hid-1.0-0.5_vgrel-7
