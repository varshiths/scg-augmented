
rel_mapper = {
    'and' : 'and',
    'says' : 'say',
    'belonging to' : 'belong to',
    'over' : 'over',
    'parked on' : 'park on',
    'growing on' : 'grow on',
    'standing on' : 'stand on',
    'made of' : 'make of',
    'attached to' : 'attach to',
    'at' : 'at',
    'in' : 'in',
    'hanging from' : 'hang from',
    'wears' : 'wears',
    'in front of' : 'front of',
    'from' : 'from',
    'for' : 'for',
    'watching' : 'watch',
    'lying on' : 'lie on',
    'to' : 'to',
    'behind' : 'behind',
    'flying in' : 'fly in',
    'looking at' : 'look at',
    'on back of' : 'on back of',
    'holding' : 'hold',
    'between' : 'between',
    'laying on' : 'lay on',
    'riding' : 'rid',
    'has' : 'have',
    'across' : 'across',
    'wearing' : 'wear',
    'walking on' : 'walk on',
    'eating' : 'eat',
    'above' : 'above',
    'part of' : 'part of',
    'walking in' : 'walk in',
    'sitting on' : 'sit on',
    'under' : 'under',
    'covered in' : 'cover in',
    'carrying' : 'carry',
    'using' : 'use',
    'along' : 'along',
    'with' : 'with',
    'on' : 'on',
    'covering' : 'cover',
    'of' : 'of',
    'against' : 'against',
    'playing' : 'play',
    'near' : 'near',
    'painted on' : 'paint on',
    'mounted on' : 'mount on',
}

def remap_preds(pred_ind_dict):
    predicate_to_ind = dict()
    for rel, ind in pred_ind_dict.items():
        if rel not in rel_mapper:
            predicate_to_ind[rel] = ind
        else:
            predicate_to_ind[rel_mapper[rel]] = ind

    assert "wear" in predicate_to_ind
    return predicate_to_ind.keys(), predicate_to_ind

test_caps = [ "jean is {} the wing".format(x) for x in rel_mapper.keys() ]

def post_process_preds(results, predicate_to_ind):
    # handle repetitions
    results[:, :, predicate_to_ind["wears"]] = results[:, :, predicate_to_ind["wear"]]
    return results

def post_process_objs(results, obj_to_ind):
    # handle repetitions
    results[:, obj_to_ind["men"], :] = results[:, obj_to_ind["man"], :]
    return results
