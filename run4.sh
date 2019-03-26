#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

python models/eval_rels_tc.py \
    -m sgdet \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-rc-1.0-0.5/vgrel-13.tar \
    -cache cache/sgdet_motifnet-size-sgdet-rc-1.0-0.5_vgrel-13

echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m sgcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-rc-1.0-0.5/vgrel-13.tar \
    -cache cache/sgcls_motifnet-size-sgdet-rc-1.0-0.5_vgrel-13

echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-rc-1.0-0.5/vgrel-13.tar \
    -cache cache/predcls_motifnet-size-sgdet-rc-1.0-0.5_vgrel-13

echo "***************************"
echo "***************************"
echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m sgdet \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-vg-1.0-0.5/vgrel-11.tar \
    -cache cache/sgdet_motifnet-size-sgdet-vg-1.0-0.5_vgrel-11

echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m sgcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-vg-1.0-0.5/vgrel-11.tar \
    -cache cache/sgcls_motifnet-size-sgdet-vg-1.0-0.5_vgrel-11

echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-size-sgdet-vg-1.0-0.5/vgrel-11.tar \
    -cache cache/predcls_motifnet-size-sgdet-vg-1.0-0.5_vgrel-11
