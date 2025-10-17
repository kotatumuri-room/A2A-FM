import torch
from torch import nn
from script.models.base_models import BaseModel
from hydra.utils import instantiate
from omegaconf import DictConfig


class SymmetricFlow(BaseModel):
    def __init__(self, mynet: DictConfig, data_handler: DictConfig, **kwargs):
        super().__init__(data_handler=data_handler, **kwargs)
        self.net = nn.ModuleList(
            [instantiate(mynet, data_handler=data_handler, _recursive_=False)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.data_handler.split_for_symmetric(z)
        f1 = self.net[0](z1)
        f2 = self.net[0](z2)
        return f2 - f1


class Flow(BaseModel):

    def __init__(self, mynet: DictConfig, data_handler: DictConfig, **kwargs):
        super().__init__(data_handler=data_handler, **kwargs)
        self.net = nn.ModuleList(
            [instantiate(mynet, data_handler=data_handler, _recursive_=False)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f1 = self.net[0](z)
        return f1
