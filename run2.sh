#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

RPATH=$(pwd)
# RPATH=$(dirname `pwd`)
export PYTHONPATH=$PYTHONPATH:$RPATH
# echo $PYTHONPATH

python models/eval_rels_tc.py \
    -m sgdet \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-tc/vgrel-9.tar \
    -cache cache/sgdet_motifnet-sgdet-tc_vgrel-9

echo "***************************"
echo "***************************"

python models/eval_rels.py \
    -m sgdet \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-nob/vgrel-9.tar \
    -cache cache/sgdet_motifnet-sgdet-nob_vgrel-9
    # -use_bias

echo "***************************"
echo "***************************"
echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m sgcls \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-tc/vgrel-9.tar \
    -cache cache/sgcls_motifnet-sgdet-tc_vgrel-9

echo "***************************"
echo "***************************"

python models/eval_rels.py \
    -m sgcls \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-nob/vgrel-9.tar \
    -cache cache/sgcls_motifnet-sgdet-nob_vgrel-9
    # -use_bias

echo "***************************"
echo "***************************"
echo "***************************"
echo "***************************"

python models/eval_rels_tc.py \
    -m predcls \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-tc/vgrel-9.tar \
    -cache cache/predcls_motifnet-sgdet-tc_vgrel-9

echo "***************************"
echo "***************************"

python models/eval_rels.py \
    -m predcls \
    -model motifnet \
    -b 8 \
    -p 100 \
    -ngpu 1 \
    -test \
    -ckpt checkpoints/motifnet-leftright-sgdet-nob/vgrel-9.tar \
    -cache cache/predcls_motifnet-sgdet-nob_vgrel-9
    # -use_bias
