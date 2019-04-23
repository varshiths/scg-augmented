#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

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
    -ckpt checkpoints/motifnet-size-predcls-nbg-nob/vgrel-5.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-nob_vgrel-5

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-vg-1.0-0.5/vgrel-5.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-vg-1.0-0.5_vgrel-5

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-coco-1.0-0.5/vgrel-6.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-coco-1.0-0.5_vgrel-6

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-hid-1.0-0.5/vgrel-9.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-hid-1.0-0.5_vgrel-9
