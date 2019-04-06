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
- motifnet-size with prior extracted from MSCOCO Captions (TODO bugfree: ?)  
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

PREDCLS for MOTIFNET-SIZE  

| Model | R@20 | R@50 | R@100 |
| ----- | ---- | ---- | ----- |
| No Bias | 0.582 | 0.649 | 0.668 |
| VG Bias (Prior) | 0.000 | 0.000 | 0.000 |
| Coco Bias (OIE) | 0.000 | 0.000 | 0.000 |
| HID Bias (OIE) | 0.000 | 0.000 | 0.000 |
| Published | 0.580 | 0.649 | 0.668 |

TODO:
- Monitor the nnumber of violations during training
- Inspect effect of prior
