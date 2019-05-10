
import sys
import json
import numpy as np
import codecs

from tqdm import tqdm
from pprint import pprint

from misc.openie_utils import remap_preds, test_caps, post_process_preds, post_process_objs

# split = sys.argv[1]
# assert split in ["a", "b", "c", "d", "e", "f"]

VG_SGG_DICT_FN = "../../datasets/VG-SGG-dicts.json"
TRIPLES_FILE = "../../datasets/hid/descriptions_svo.txt"

INTM_FILE = "mscoco_captions_text.npy"
OUTPUT_FILE = "mscoco_captions_freq.npy"

# INTM_FILE = "descriptions_text.npy"
# OUTPUT_FILE = "descriptions_freq.npy"

# CAPTIONS_FILE = "../../datasets/hid/split_descriptions_{}.txt".format(split)
# OUTPUT_FILE = "descriptions_freq_a{}.npy".format(split)

def load_info(info_file):
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0
    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    return class_to_ind, predicate_to_ind

class_to_ind, predicate_to_ind = load_info(VG_SGG_DICT_FN)

objs = class_to_ind.keys()
# opreds = predicate_to_ind.keys()
preds, predicate_to_ind = remap_preds(predicate_to_ind)

num_classes = len(objs)
num_predicates = len(preds)

# triples = []
# print("Reading file...")
# with codecs.open(TRIPLES_FILE) as f:
#     for line in f:
#         svo = [x.strip() for x in line.split("|")]
#         if len(svo) != 3:
#             continue
#         triples.extend(svo)
# print("Done.")

# import spacy
# spacy.prefer_gpu()

# nlp = spacy.load('en', disable=['parser', 'ner'])

# _triples = triples

# ltriples = []
# for doc in tqdm(nlp.pipe(
#     texts=_triples, 
#     as_tuples=False, 
#     n_threads=2, 
#     batch_size=10000), total=len(_triples)):    
#     ltriples.append([x.lemma_ for x in doc])
# ltriples = np.array(ltriples)

# ltriples = np.reshape(ltriples, (-1, 3))
# np.save(INTM_FILE, ltriples)

ltriples = np.load(INTM_FILE)

def ret_ind(objc, dt):
    for obj, ind in dt.items():
        if obj in objc:
            return ind, obj
    return None, None

def ret_ind_pred(predc, dt):
    for pred, ind in dt.items():
        if pred in " ".join(predc):
        # if " ".join(predc) in pred:
            return ind, pred
    return None, None

grels = np.zeros((
    num_classes,
    num_classes,
    num_predicates,
), dtype=np.int64)

for triple in tqdm(ltriples):
    sind, s = ret_ind(triple[0], class_to_ind)
    oind, o = ret_ind(triple[2], class_to_ind)
    pind, p = ret_ind_pred(triple[1], predicate_to_ind)
    if sind and oind and pind:
        # if p == "cover in" and s == "man":
        #     print(triple)
        # print(triple)
        grels[sind, oind, pind] += 1

grels = post_process_preds(grels, predicate_to_ind)
grels = post_process_objs(grels, class_to_ind)

np.save(OUTPUT_FILE, grels)
