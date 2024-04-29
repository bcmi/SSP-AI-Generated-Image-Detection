import torch
from torch import nn
import torch.nn.functional as F


def bceLoss():
    return nn.BCEWithLogitsLoss()


def crossEntropyLoss():
    return nn.CrossEntropyLoss()
def mseLoss():
    return nn.MSELoss()