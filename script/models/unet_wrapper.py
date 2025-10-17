import torch
from script.data_handler.data_handler import DataHandler
from script.models.base_models import BaseModel
from torch import nn
import pickle
from hydra.utils import instantiate
from script.models.unet.unet import UNetModelWrapper
from omegaconf import DictConfig


class UNetWrapper(BaseModel):
    def __init__(self, unet_model: DictConfig, **kwargs):
        super().__init__(**kwargs)
        unet_model = instantiate(
            unet_model, continous_condition=self.data_handler.model_condition_dim
        )
        assert isinstance(
            unet_model, UNetModelWrapper
        ), f"model has to be UNetModel got {unet_model}"
        self.net = nn.ModuleList([unet_model])
        self.pre_layer = nn.Linear(
            self.data_handler.model_condition_dim,
            1,
        )
        self.image_size = unet_model.image_size
        self.in_channels = unet_model.in_channels

    def forward(self, z):
        x, c, t = self.data_handler.expand_model_input(z, self.pre_layer)
        t = t.squeeze(-1)
        batch_size = len(x)
        x = x.reshape(batch_size, self.in_channels, self.image_size, self.image_size)
        return self.net[0](t, x, c.squeeze(1)).reshape(
            batch_size, self.in_channels * self.image_size * self.image_size
        )
