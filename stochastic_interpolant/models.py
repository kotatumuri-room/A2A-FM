from torch.nn import Module
from torch import nn
from hydra.utils import instantiate
import torch
from omegaconf import DictConfig
from coatiLDM.models.score_models.non_conv_unet import NonConvUNet
from script.models.unet.unet import UNetModelWrapper
import pickle

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


class VelocityFieldModel(Module):
    def __init__(self, model: DictConfig, K: int):
        super().__init__()
        x_dim = model.model_output_dim
        model_input = model.model_input_dim
        input_dim = K + x_dim
        self.prelayer = nn.Linear(input_dim, model_input)
        self.model = instantiate(model, _recursive_=False)

    def forward(self, alpha, x):
        z = torch.cat([alpha, x], dim=-1)
        return self.model(self.prelayer(z))


class SinAlphaModel(Module):
    def __init__(self, N: int, K: int, i: int, j: int, dt: float = 1e-3):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(K, N))
        self.i = i
        self.j = j
        self.N = N
        self.dt = dt

    def forward(self, t):
        kernels = torch.stack(
            [torch.sin(t * torch.pi * n) for n in range(self.N)], dim=1
        ).to(self.a.device)
        alpha = torch.einsum("bn, kn->bk", (kernels, self.a)) ** 2
        alpha[:, self.i] = alpha[:, self.i] + 1 - t
        alpha[:, self.j] = alpha[:, self.j] + t
        return alpha / torch.sum(alpha, dim=1).unsqueeze(1)

    def dot(self, t):
        return (self(t + self.dt) - self(t)) / self.dt


class DeepDenseNet(Module):
    def __init__(
        self,
        w=64,
        activation="SELU",
        depth=5,
        normalize=False,
        model_input_dim=1,
        model_output_dim=2,
    ):
        super().__init__()
        self.input_dim = model_input_dim
        self.output_dim = model_output_dim
        self.activation = activation_functions[activation]
        self.layers = torch.nn.ModuleList()
        self.normlayers = torch.nn.ModuleList()
        for i in range(depth):
            layer = torch.nn.Linear(self.input_dim + w * i, w)
            self.layers.append(layer)
            if normalize:
                self.normlayers.append(nn.BatchNorm1d(self.input_dim + w * (i + 1)))

        self.final_layer = torch.nn.Linear(self.input_dim + w * depth, self.output_dim)

    def forward(self, z: torch.Tensor):
        for k, layer in enumerate(self.layers):
            z = torch.cat((z, layer(z)), dim=-1)
            if hasattr(self, "normlayers"):
                if len(self.normlayers) == len(self.layers):
                    z = self.normlayers[k](z)
            z = self.activation(z)
        z = self.final_layer(z)
        return z


class UnetModelWrapper(Module):

    def __init__(self, unet_model: DictConfig, K: int, **kwargs):
        super().__init__()
        unet_model = instantiate(unet_model, continous_condition=K)
        assert isinstance(
            unet_model, UNetModelWrapper
        ), f"model has to be UNetModel got {unet_model}"
        self.net = nn.ModuleList([unet_model])
        self.image_size = unet_model.image_size
        self.in_channels = unet_model.in_channels

    def forward(self, alpha, x):
        requires_grad = next(self.parameters().__iter__()).requires_grad
        t = torch.zeros(len(x), device=x.device, requires_grad=requires_grad)
        batch_size = len(x)
        x = x.reshape(batch_size, self.in_channels, self.image_size, self.image_size)
        return self.net[0](t, x, alpha).reshape(
            batch_size, self.in_channels * self.image_size * self.image_size
        )


class COATI_wrapper(Module):
    def __init__(
        self, ckpt_doc, use_state_dict: bool, K: int, time_max: float = 1.0, **kwargs
    ):
        super().__init__()
        with open(ckpt_doc, "rb") as f_in:
            model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
        model_kwargs = model_doc["score_model_params"]
        model_kwargs["time_max"] = time_max
        model_kwargs["cond_dim"] = K
        self.net = nn.ModuleList([NonConvUNet(**model_kwargs)])
        state_dict = model_doc["model"]
        self.pre_layer = torch.nn.Linear(
            K,
            1,
        )
        if use_state_dict:
            self.net.load_state_dict(state_dict)

    def forward(self, alpha, x):
        t = self.pre_layer(alpha)
        b, d = x.shape
        return self.net[0](x, t.squeeze(1), alpha).reshape(b, d)
