#!/usr/bin/bash

MDIR=`pwd`

# setting defaults
CORPUS_FILE=""
SVO_FILE=/tmp/svo.txt
LEMMA_FILE=/tmp/lemma.npy
PRIOR_FILE=""

# argument parsing
while getopts ":c:s:l:p:h" opt; do
  case ${opt} in
	c )
	  CORPUS_FILE=`realpath $OPTARG`
	  ;;
	s )
	  SVO_FILE=`realpath $OPTARG`
	  ;;
	l )
	  LEMMA_FILE=`realpath $OPTARG`
	  ;;
	p )
	  PRIOR_FILE=`realpath $OPTARG`
	  ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
    h )
      echo "Usage: ./main.sh -c corpus_file -p prior_file [-s svo_file] [-l lemma_file]" 1>&2
      echo ""
      echo "    corpus_file	: file containing captions; should have one caption per line ending with a period."
      echo "    prior_file	: file into which prior is written in .npy format. Must have .npy suffix"
      echo "    svo_file	: intermediate file for svo triples extracted from captions."
      echo "    lemma_file	: intermediate file for lemmatized svo triples extracted from captions stored in .npy format. Must have .npy suffix"
      echo ""
      exit 1
      ;;
        
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# enforcing some conditions
if [[ $CORPUS_FILE == "" ]]; then
	echo "Error: Input corpus -c file not provided" 1>&2
	exit 1
fi
if [[ $PRIOR_FILE == "" ]]; then
	echo "Error: Output prior -p file not provided" 1>&2
	exit 1
fi

# print all arguments for confirmation
echo "CORPUS_FILE" $CORPUS_FILE
echo "SVO_FILE" $SVO_FILE
echo "LEMMA_FILE" $LEMMA_FILE
echo "PRIOR_FILE" $PRIOR_FILE

echo "-------------------------------------------------------"

# Generation of svo triples
if [ -f $SVO_FILE ]; then
	echo "Loading SVO Triples from file $SVO_FILE."
else
	cd $MDIR/misc
	./process_large_corpus.sh $CORPUS_FILE $SVO_FILE
	cd $MDIR
fi

echo "-------------------------------------------------------"

# Generation of prior from svo triples
if [ -f $PRIOR_FILE ]; then
	echo "Prior $PRIOR_FILE already exists on disk."
else
	python openie_freq.py $SVO_FILE $LEMMA_FILE $PRIOR_FILE
fi

echo "-------------------------------------------------------"

echo "Training the model with prior from corpus"

export CUDA_VISIBLE_DEVICES=2
RPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$RPATH

python models/train_rels_tc.py \
    -m predcls \
    -model motifnet \
    -order size \
    -b 8 \
    -p 1000 \
    -ngpu 1 \
    -ckpt checkpoints/vgdet/vg-24.tar \
    -save_dir checkpoints/motifnet-size-predcls-nobg-nob
