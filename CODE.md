# Code Base Guide

## Folder Structure

The code base is divided into four main folders:  

1. `dataloaders`
1. `lib`
1. `models`
1. `misc`

### `dataloaders`

As the name suggests, dataloaders contains the code for loading of images from a directory of images, graphs from a json file; preprocessing in train/test scenarios.

The code relevant to Visual Genome is contained in the file `dataloaders/visual_genome.py`. Each of the functions are documented.

### `lib`

This folder contains the code for all the models involved in the experiments along with code relevant for evaluation. Most of the names of the files/folders are self-explanatory.

1. `lib/object_detector.py`: Contains the code for the object detection network `ObjectDetector` used in the paper.
1. `lib/rel_model.py`: Contains the code for the base model MOTIFNET, `RelModel`.
1. `lib/rel_model_prior.py`: Contains the code for a network which is MOTIFNET with the prior incorporated, `RelModelPrior`. This is useful if the distillation experiment requires a passive teacher that does not learn.
1. `lib/rel_model_tc.py`: Contains code for a teacher student setup `RelModelTC` where the teacher is the student network incorporated with prior. The teacher and student networks here share parameters which are updated after every batch.
1. `lib/sparse_targets.py`: Contains code for either building prior from dataset (in the case of VG-Bias) or loading prior from npy files (in case of COCO Captions / HID). The generation of these priors are not folded into the code to keep modules independent of each other.

### `models`

This folder contains code for training and evaluating the models in various settings.

1. `models/eval_rel_count.py`: Evaluation of the baseline approach that makes predictions based on the relation counts in the dataset.
1. `models/eval_rels.py`: Evaluates a MOTIFNET model on the validation / train datasets.

1. `models/train_detector.py`: Contains code for training an `ObjectDetector` network.
1. `models/train_rels.py`: Contains code for training a `RelModel`.
1. `models/train_rels_tc.py`: Contains code for training `RelModelTC` in a student-teacher distillation framwork where the student and teacher share parameters.
1. `models/train_rels_it.py`: Contains code for training `RelModel` in a student-teacher distillation framework were the teacher is built from a pretrained network and is non-trainable.
1. `models/train_rels_ac.py`: Contains code for training `RelModelTC` in a shared student-teacher distillation framework in an active learning setting with a random acquisition criterion.

### `misc`

This folder contains code relevant for preprocessing captions in a text file and generation of a prior for the sake of use.

1. `misc/process_large_corpus.sh`: Script that takes in a corpus file as input. The corpus should be one sentence in a line ending with a period. This is to be run from inside the misc folder. This produces a file with svo triples.
All other files are relevant to working of this openie parser.

### root

The root folder contains the following scripts.

1. `openie_freq.py`: Takes as input the svo triples file and outputs an `.npy` file. This file is the prior counts as extracted from the svo triples. Set file names at the beginning of the file.
1. `config.py`: Contains the list of arguments various files in `models` folder take. Also is the location of all paths of various files.
1. `run*.sh`: Sample runs of training scripts.
