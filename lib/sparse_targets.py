from lib.word_vectors import obj_edge_vectors
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from config import DATA_PATH
import os
from lib.get_dataset_counts import get_counts


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, eps=1e-3):
        super(FrequencyBias, self).__init__()
        self.init_fg_matrix()
        self.build(eps=eps)

    def init_fg_matrix(self):
        fg_matrix, bg_matrix = get_counts(must_overlap=True)
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        self.fg_matrix = fg_matrix

    def build(self, eps=1e-3):
        fg_matrix = self.fg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline


class RCCorpusBias(FrequencyBias):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img) from a corpus.
    This class inherits from the FrequencyBias class and alters the 
    fg_matrix init to load from saved pickle file
    """
    def __init__(self, corpus=None, eps=1e-3):
        super(RCCorpusBias, self).__init__(eps)

    def init_fg_matrix(self, corpus="region_captions"):
        try:
            self.fg_matrix = np.load("data/captions_freq.npy") + 1
        except Exception as e:
            raise Exception("Please generate captions_freq.npy using scripts in misc and then proceed")

class COCOCorpusBias(FrequencyBias):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img) from a corpus.
    This class inherits from the FrequencyBias class and alters the 
    fg_matrix init to load from saved pickle file
    """
    def __init__(self, corpus=None, eps=1e-3):
        super(COCOCorpusBias, self).__init__(eps)

    def init_fg_matrix(self, corpus="mscoco_captions"):
        try:
            self.fg_matrix = np.load("data/mscoco_captions_freq.npy") + 1
        except Exception as e:
            raise Exception("Please generate mscoco_captions_freq.npy using scripts in misc and then proceed")


if __name__ == '__main__':
    fqb = CorpusBias()
    import pdb; pdb.set_trace()
