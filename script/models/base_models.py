import torch
from script.data_handler.data_handler import DataHandler
import numpy as np
from torch import nn
import torch.nn.functional as F

activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.01),
    "ELU": nn.ELU(alpha=1.0),
    "PReLU": nn.PReLU(num_parameters=1, init=0.25),
    "Mish": nn.Mish(),
    "SELU": nn.SELU(),
    "Hardshrink": nn.Hardshrink(),
}


class BaseModel(nn.Module):
    def __init__(self, data_handler: DataHandler):
        super().__init__()
        self.data_handler = data_handler

    def forward(self, z: torch.Tensor):
        raise NotImplementedError


class DeepDenseNet(BaseModel):
    def __init__(self, w=64, activation="SELU", depth=5, normalize=False, **kwargs):
        super().__init__(**kwargs)
        input_dim = self.data_handler.model_input_dim
        output_dim = self.data_handler.model_output_dim
        self.activation = activation_functions[activation]
        self.layers = torch.nn.ModuleList()
        self.normlayers = torch.nn.ModuleList()
        for i in range(depth):
            layer = torch.nn.Linear(input_dim + w * i, w)
            self.layers.append(layer)
            if normalize:
                self.normlayers.append(nn.BatchNorm1d(input_dim + w * (i + 1)))

        self.final_layer = torch.nn.Linear(input_dim + w * depth, output_dim)

    def forward(self, z: torch.Tensor):
        for k, layer in enumerate(self.layers):
            z = torch.cat((z, layer(z)), dim=-1)
            if hasattr(self, "normlayers"):
                if len(self.normlayers) == len(self.layers):
                    z = self.normlayers[k](z)
            z = self.activation(z)
        z = self.final_layer(z)
        return z
