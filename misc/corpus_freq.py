'''
To be run from root folder as misc/corpus_freq.py
Generates the biases from given corpus and places in data folder

uses spacy backend.

to setup:
pip install -U spacy
python -m spacy download en
'''

import json
import numpy as np
import sng_parser
import codecs

from tqdm import tqdm

def load_info(info_file):
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0
    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    return class_to_ind, predicate_to_ind

VG_SGG_DICT_FN = "data/stanford_filtered/VG-SGG-dicts.json"
class_to_ind, predicate_to_ind = load_info(VG_SGG_DICT_FN)

objs = class_to_ind.keys()
preds = predicate_to_ind.keys()

def get_tuples(cap):
    graph = sng_parser.parse(cap)
    entities = graph["entities"]
    relations = graph["relations"]
    tups = []
    for relation in relations:
        tups.append((
            entities[relation["subject"]]["lemma_head"],
            relation["lemma_relation"], 
            entities[relation["object"]]["lemma_head"],
            ))
    del graph
    del entities
    del relations
    return tups

num_classes = len(objs)
num_predicates = len(preds)

# corpus is one sentence in each line
with codecs.open("data/visgenome/captions.txt") as f:
    print("Reading file...")
    caps = [ x.strip() for x in f.readlines() ]

# from joblib import Parallel, delayed
import multiprocessing
import math

def batch_iterate(caps, nthreads):
    nitems = len(caps)
    batch_size = math.ceil(nitems / nthreads)
    for i in range(nthreads):
        yield caps[ i*batch_size : (i+1)*batch_size ]

def myfunc(batch_caps):
    grels = np.zeros((
        num_classes,
        num_classes,
        num_predicates,
    ), dtype=np.int64)
    for i, cap in enumerate(tqdm(batch_caps)):
        # print("{}: {}".format(i, cap))
        tups = get_tuples(cap)
        for s, r, o in tups:
            if r in preds and s in objs and o in objs:
                grels[ class_to_ind[s], class_to_ind[o], predicate_to_ind[r] ] += 1
    return grels

num_cores = multiprocessing.cpu_count()
# num_cores = 2
pool = multiprocessing.Pool(processes=num_cores)
results = sum(pool.map( myfunc, batch_iterate(caps, nthreads=num_cores) ))
# results = myfunc(caps)

np.save("data/captions_freq.npy", results)
