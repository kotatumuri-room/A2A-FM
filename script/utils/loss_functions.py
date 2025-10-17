from torch.nn import Module
import pickle
import torch
from torch.nn.functional import mse_loss
import numpy as np



class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, **kwargs):
        return mse_loss(x, y)
