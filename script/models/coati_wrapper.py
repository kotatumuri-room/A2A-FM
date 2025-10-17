import torch
from script.data_handler.data_handler import DataHandler
from script.models.base_models import BaseModel
from torch import nn
import pickle

from coatiLDM.models.score_models.non_conv_unet import NonConvUNet


class COATI_wrapper(BaseModel):
    def __init__(self, ckpt_doc, use_state_dict: bool, time_max: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        with open(ckpt_doc, "rb") as f_in:
            model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
        model_kwargs = model_doc["score_model_params"]
        model_kwargs["time_max"] = time_max
        model_kwargs["cond_dim"] = self.data_handler.model_condition_dim
        self.net = nn.ModuleList([NonConvUNet(**model_kwargs)])
        state_dict = model_doc["model"]
        self.pre_layer = torch.nn.Linear(
            self.data_handler.model_condition_dim,
            1,
        )
        if use_state_dict:
            self.net.load_state_dict(state_dict)

    def forward(self, z):
        x, c, t = self.data_handler.expand_model_input(z, self.pre_layer)
        t = t.squeeze(-1)
        return self.net[0](x, t, c)
