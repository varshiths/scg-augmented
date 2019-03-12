# scg-augmented

This work attempts to implement knowledge distillation ([Visual Relationship Detection with Internal and External Linguistic Knowledge Distillation](https://arxiv.org/pdf/1707.09423.pdf)) over neural motifs ([Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)](https://arxiv.org/abs/1711.06640v2)).

This is the first step of a more daring attempt to build a model that constructs scene graphs without any gold scene graph data that is rather difficult to acquire.

## 6th - 13th March

Implemented knowledge distillation of prior information obtained from counts of occurrences of a particular relationship for an object predicate pair.  

Results were comparable to that of a neural-motifs model that directly alters the softmax output with the same prior information proving that KL-distillation is effective. The parameters involved in the distillation require more tuning.

Trained / Biased Pretrained  
R@20: 0.317 / 0.322  
R@50: 0.347 / 0.350  
R@100: 0.355 / 0.357  

