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
    -ckpt checkpoints/motifnet-size-predcls-nbg-3-nob/vgrel-5.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-3-nob_vgrel-5

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-3-coco-1.0-0.5/vgrel-12.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-3-coco-1.0-0.5_vgrel-12

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-3-vg-1.0-0.5/vgrel-10.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-3-vg-1.0-0.5_vgrel-10

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 16 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-predcls-nbg-3-hid-1.0-0.5/vgrel-5.tar \
    -cache cache/sgdet_motifnet-size-predcls-nbg-3-hid-1.0-0.5_vgrel-5
