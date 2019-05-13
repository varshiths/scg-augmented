# scg-augmented

This work attempts to implement knowledge distillation ([Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation](https://arxiv.org/pdf/1707.09423.pdf)) over neural motifs ([Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)](https://arxiv.org/abs/1711.06640v2)).

This is the first step of a more daring attempt to build a model that constructs scene graphs without any gold scene graph data that is rather difficult to acquire.

## 6th - 13th March

Implemented knowledge distillation of prior information obtained from counts of occurrences of a particular relationship for an object predicate pair.  

Results were comparable to that of a neural-motifs model that directly alters the softmax output with the same prior information proving that KL-distillation is effective. The parameters involved in the distillation require more tuning.


SGCLS for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| Trained | 0.317 | 0.347 | 0.355 |
| Published | 0.322 | 0.350 | 0.357 |


Left-Right ordering of objects is optimal. Freezing the choice of the order.  

Completed pipeline setup for relations prior extraction from corpus. Added file in misc.

## 14th - 20th March

Processing captions
abcdef
vvvvvv
svhvsh

Training currently and evaluated:  

1. sdget with no bias  
2. sgdet with Visual Genome Prior bias  
3. sdget with bias built from region captions corpus

SGDET for MOTIFNET-LEFTRIGHT

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.211 | 0.269 | 0.301 |
| VG Bias | 0.209 | 0.266 | 0.297 |
| RC Bias | 0.000 | 0.000 | 0.000 |

SGCLS for MOTIFNET-LEFTRIGHT

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.199 | 0.265 | 0.302 |
| VG Bias | 0.129 | 0.209 | 0.268 |
| RC Bias | 0.000 | 0.000 | 0.000 |

PREDCLS for MOTIFNET-LEFTRIGHT

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.256 | 0.413 | 0.535 |
| VG Bias | 0.149 | 0.286 | 0.423 |
| RC Bias | 0.000 | 0.000 | 0.000 |



Is it a good idea to have a soft loss for teacher student predictions?  
No apparently. Also the performance has fallen a lot due to this (suspiciously only for leftright ordering though).  

The hyperparameters also seem problematic. Need to verify that the problem is because of soft teacher loss and not hyperparams.  

## 21th - 28th March

Currently, have changed the soft preds of teacher to hard preds with the same hyperparams for config-size as that seemed to have worked.  

Being trained:
- vg-bias  
- rc-bias  
- no-bias (TODO)  

To be done later:
`prior_weight` to be decreased to 0.2 for vg-bias to make teacher_dists range comparable with that of rel_dists and avoid an overpovering prior.  

Improved prior extraction by matching names of rels and extracted relations.
- hid (changed)
- rc (TODO)

Descriptions Corpus:  
Is the descriptions corpus informative enough?  
- (Not seemingly)  

To check:  
Inspect the prior of VG beside RC and HID  
- (possibly distribution is better represented due to exact relation labels)  

SGDET for MOTIFNET-SIZE

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.208 | 0.267 | 0.298 | 
| VG Bias | 0.209 | 0.265 | 0.297 | 
| RC Bias | 0.111 | 0.150 | 0.178 | 
| Published | 0.216 | 0.273 | 0.304 | 

To check:
Is the training of a model mode dependent as well?
- Yes, it is. Model is to be trained and evaluated in each of the modes separately.  

## 29th March - 3rd April

Shift to OpenIE parser?  
- The sng_parser which was rule based relied on a closed set of verb phrases to form relation tuples which was obviously wrong  

Used the OpenIE parser to extract tuples from the following corpora  
- MSCOCO Image Captions
- Hierarchical Image Descriptions

Under training:  
- motifnet-size with prior extracted from MSCOCO Captions (TODO bugfree: done)  
- motifnet-size with prior extracted from HID (TODO: skip)  

Shift to PREDCLS?  
- Probably should. Training SGDET takes considerably longer and the results are not indicative of the prior performance?  

SGDET for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.208 | 0.267 | 0.298 | 
| VG Bias (Prior) | 0.209 | 0.265 | 0.297 | 
| RC Bias (SGP) | 0.111 | 0.150 | 0.178 | 
| Coco Bias (OIE) | 0.091 | 0.127 | 0.156 | 
| Published | 0.216 | 0.273 | 0.304 | 

OIE: OpenIE based tuple extraction  
SGP: Spacy Scene Graph Parser at github.com/sng_parser  

Insights:  
- Bad hyperparameter tuning  
- Shift to PREDCLS to perform quick experimentation  
- Prior for negative cases?  

## 4th - 10th April

Under training:  
- ~~motifnet-size-sgdet with prior extracted from MSCOCO Captions~~
- motifnet-size-predcls nob, vgp, todo: coco and hid

To establish:
- lk distillation works well with what params of vgp
- extend/repeat with other params

TODO:
- Monitor the nnumber of violations during training
- Inspect effect of prior

PREDCLS for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.582 | 0.649 | 0.668 |
| VG Bias (Prior) | 0.568 | 0.637 | 0.656 |
| Coco Bias (OIE) | 0.280 | 0.372 | 0.434 |
| HID Bias (OIE) | 0.461 | 0.561 | 0.611 |
| Published | 0.580 | 0.649 | 0.668 |

HPARAM EXPLR - VAL - PREDCLS for MOTIFNET-SIZE  

| Coco Bias | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| 1.0, 0.5\* | 0.234 | 0.305 | 0.360 |
| 0.8, 0.3 | 0.366 | 0.440 | 0.481 |
| 0.8, 0.5 | 0.282 | 0.356 | 0.404 |
| 0.0, 0.0 | 0.606 | 0.659 | 0.675 |

| HID Bias | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| 1.0, 0.5 | 0.473 | 0.558 | 0.603 |

To avoid bad predictions in the beginning, only the student network is trained for the first five epochs.  
This is simulated by starting training from the nob checkpoint at epoch 5.

Insights:
- Monitored the softmax values during training
- Problem still persists - check hyper params exploration on validation set
- The number of preds \ bg pred correctly is better by teacher
- Teacher almost always misses “bg” - student is better
- The pressure for student to match “bg” is hurting the scores of the other relations
- The VG Bias (Prior) accounts for overlaps, i.e., if boxes overlap, then the relation is "bg"

Approaches:
- Figure out if "bg" class can be skipped from the teaching process
- Manually construct bg from language (difficult)
- Decrease the probability of bg class globally

## 11th - 17th April

Because the frequency of "bg" is high, the model requires you to be good with the other classes too.  

Not just about training with less weight for "bg", it is also ensuring the rest of the confidence values are also in order.

- Use ranking losses?  
- No clue how the model with the good prior actually does this.

Attempt with class weighting for teacher-student loss term:  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| Coco Bias | 0.280 | 0.372 | 0.434 |
| Coco Bias + ICR | 0.198 | 0.275 | 0.330 |
| HID Bias | 0.461 | 0.561 | 0.611 |
| HID Bias + ICR | 0.408 | 0.500 | 0.549 |
| Published | 0.580 | 0.649 | 0.668 |

TODO:
- Train the models ignoring the "bg" class to inspect the effect of the prior.

## 18th - 24th April

Currently training:
- Models with "bg" class masked out in both the teacher loss term and the gold loss term.

PREDCLS for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | ~~0.368~~ | ~~0.522~~ | ~~0.612~~ |
| VG Bias | ~~0.341~~ | ~~0.471~~ | ~~0.551~~ |
| COCO Bias | ~~0.394~~ | ~~0.515~~ | ~~0.577~~ |
| HID Bias | ~~0.302~~ | ~~0.448~~ | ~~0.540~~ |
| Published | ~~0.580~~ | ~~0.649~~ | ~~0.668~~ |

**Note: Table invalid. Updated table after bug fix. In section 2 May to 10th May.**

- More hyper parameter tuning?
- Look at images and priors

Insights:
- Bad performace of HID is probably a hyperparam issue. The prior is weaker in case of HID, with the max value of prior confidence being 0.04 less than that of coco. Also, distillation works better with coco than hid (observed qualitatively). Find better ways to tune.
- At low threshold, recall is bad as the confidence values are lower in general due to absence of training with negative label. In need of negative label / other methods to boost model confidence and decrease entropy. Train more?

## 25th April - 1st May

Recall@N explained:  
Say there are P candidate obj-pairs for an image. For the P pairs, predict the most likely  
predicate. This pair has a score:  
predicate_score * obj_score1 * obj_score2  
Consider the top N predictions of the model as per this score.  
Among these, measure recall of the ground_truth relations among the valid pairs.  
The score is the average of this score for various images in the test set.  

Predicate scores and predictions exclude the "bg" class.  

Number of object pairs where "bg" had the highest confidence: \~10 among 2000000 candidate predictions.  
Even after inclusion in the predicate scores and predictions, the performance of a model trained with COCO bias did not change.   
This says that bg is not the most confident prediction but is relevant in balancing other logits and assigning high confidence to correct classes.  

Insights:  
This is not an issue with bg class calibration. "bg" has a very low score.  
Is this that the test distribution and the prior distribution is very different?  

TODO:
- To deal with the bg class problem and enforce high confidence misbalance.

Currently training:
- Models with "bg" class masked out in teacher loss term, gold loss term with 30000 the images in the dataset, to inspect if prior helps bridge the gap between train and test.  

PREDCLS for MOTIFNET-SIZE  

| Model | Fraction | R@20 | R@50 | R@100 |
| ----- | ----- | ---- | ---- | ----- |
| No Bias | 100% | 0.368 | 0.522 | 0.612 |
| No Bias |  30% | 0.344 | 0.488 | 0.581 |
| VG Bias | 100% | 0.341 | 0.471 | 0.551 |
| VG Bias |  30% | 0.205 | 0.330 | 0.437 |
| COCO Bias | 100% | 0.394 | 0.515 | 0.577 |
| COCO Bias |  30% | 0.427 | 0.532 | 0.578 |
| HID Bias | 100% | 0.302 | 0.448 | 0.540 |
| HID Bias |  30% | 0.209 | 0.334 | 0.444 |
| Published | - | 0.580 | 0.649 | 0.668 |

**Note: This table is invalid. To update table after bug fix.**

- HID is on a subset of VG. The fall in performance is possibly because of that.
- Why VG Bias falls so much in performance with limited data as compared to No-Bias is not clear as well.

## 7th May - 1st May

Concept Net Summary:
- contains all objects
- contains the following relations

| ----- | ----- | ---- |
| RelatedTo | HasPrerequisite | Entails |
| FormOf | HasProperty | MannerOf |
| IsA | MotivatedByGoal | LocatedNear |
| PartOf | ObstructedBy | HasContext |
| HasA | Desires | dbpedia |
| UsedFor | CreatedBy | SimilarTo |
| CapableOf | Synonym | EtymologicallyRelatedTo |
| AtLocation | Antonym | EtymologicallyDerivedFrom |
| Causes | DistinctFrom | CausesDesire |
| HasSubevent | DerivedFrom | MadeOf |
| HasFirstSubevent | SymbolOf | ReceivesAction |
| HasLastSubevent | DefinedAs | InstanceOf |
| ----- | ----- | ---- |

- can build containment, prepositional, typeof, with relations
- verbs exist as nodes but do not exhaustively list all objects involved in the action
- verbs do not exist as relations
- can harvest RelatedTo, HasSubevent, relation but with low confidence

Building KGs:
- Dictionary examples of verb phrases typically list down most possible uses of a phrase
- Is possible to fine type the example sentences and replace with related objects from word net

## 2nd May - 10th May

Discovered a bug where the teacher preds were off my 1.
To fix and repeat the last two experiments.

Model with the background class masked out for both the teacher loss as well as the gold label loss.  

PREDCLS for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.354 | 0.509 | 0.602 |
| VG Bias | 0.128 | 0.259 | 0.399 |
| COCO Bias | 0.117 | 0.238 | 0.369 |
| HID Bias | 0.122 | 0.247 | 0.385 |
| Published | 0.580 | 0.649 | 0.668 |

It is obvious that no kind of prior seems to be helping. One of possible reasons why in case of the bug, the results were coherent could be because  
the teacher loss term is uniform noise. In this case, the model probably ignored the teacher loss and optimized the gold loss.  
It is not very clear if this is actually the case.  

However, one possible reason why the model is underperforming is because there could be a possible feedback effect.  
If one particular logit, as predicted by the student model is already strong and the prior supports it,  
it is likely to be strengthened further.

Since there is a leakage class for the classifier, there is no natural bound to the logits and so it is possible there is a shift in the mean of the logits.

This particular distribution shift does happen in the case of a classifier with no examples for one of the classes, which is what is happening with the mask.
