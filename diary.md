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
