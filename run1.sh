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
    -ckpt checkpoints/motifnet-leftright-sgdet-tc/vgrel-8.tar \
    -cache cache/motifnet_sgdet

# python models/eval_rels.py \
#     -m predcls \
#     -model motifnet \
#     -order leftright \
#     -nl_obj 2 \
#     -nl_edge 4 \
#     -b 6 \
#     -clip 5 \
#     -p 100 \
#     -hidden_dim 512 \
#     -pooling_dim 4096 \
#     -lr 1e-3 \
#     -ngpu 1 \
#     -test \
#     -ckpt checkpoints/motifnet-sgcls/vgrel-7.tar \
#     -nepoch 50 \
#     -use_bias \
#     -cache motifnet_predcls
