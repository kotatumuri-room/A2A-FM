from abc import ABCMeta, abstractmethod
from typing import Tuple
import torch
from omegaconf import DictConfig
from script.data_handler.data_handler import DataHandler


class BetaStyleTransferDataHandler(DataHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_dim = None
        self.model_output_dim = None

    def get_x(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        assert z.shape[1] == self.x_dim + self.c_dim
        return z[:, : self.x_dim]

    def get_c(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        assert z.shape[1] == self.x_dim + self.c_dim
        return z[:, self.x_dim :]

    def reshape_batch(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(x) == len(c)
        assert x.dim() == 2 and c.dim() == 2
        return torch.cat([x, c], dim=1)

    def convert_target(self, zt: torch.Tensor) -> torch.Tensor:
        return self.get_x(zt)


class A2AFMDataHandler(BetaStyleTransferDataHandler):
    """this class demands symmetric flow"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_dim = self.x_dim + self.c_dim * 2
        self.model_condition_dim = self.c_dim * 2
        self.model_output_dim = self.x_dim

    def ot_variables(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.get_x(z0)
        c0 = self.get_c(z0)
        x1 = self.get_x(z1)
        c1 = self.get_c(z1)

        return torch.cat([x0, c0, c1], dim=1), torch.cat([x1, c0, c1], dim=1)

    def apply_beta(self, z_hat: torch.Tensor, beta: float) -> torch.Tensor:
        x, c0, c1 = z_hat.split([self.x_dim, self.c_dim, self.c_dim], dim=1)
        return torch.cat([x, c0 * beta, c1 * beta], dim=1)

    def model_input_terminals(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(z0) == len(z1)
        assert z0.dim() == z1.dim() == 2
        x0 = self.get_x(z0)
        c0 = self.get_c(z0)
        x1 = self.get_x(z1)
        c1 = self.get_c(z1)
        return torch.cat([x0, c0, c0, c1], dim=1), torch.cat([x1, c1, c0, c1], dim=1)

    def expand_model_input(
        self, z: torch.tensor, pre_layer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, c = z.split([self.x_dim, self.model_condition_dim], dim=1)
        t = pre_layer(c)
        return x, c, t

    def t_override_model_input(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, ct, c = z.split([self.x_dim, self.c_dim, self.c_dim], dim=1)
        return x, c, ct

    def split_for_symmetric(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xt, ct, c0, c1 = z.split(
            [self.x_dim, self.c_dim, self.c_dim, self.c_dim], dim=1
        )
        z0 = torch.cat([xt, ct, c0], dim=1)
        z1 = torch.cat([xt, ct, c1], dim=1)
        return z0, z1

    def inference_model_input(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        initc: torch.Tensor,
        targc: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            initc.shape == targc.shape
        ), f"shapes of conditions are different {initc.shape}, {targc.shape}"
        ct = initc * (1 - t) + targc * t
        return torch.cat([xt, ct, initc, targc], dim=1)

class A2AFMDataHandlerNoSim(A2AFMDataHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_dim = self.x_dim + self.c_dim * 3
        self.model_condition_dim = self.c_dim * 3
        self.model_output_dim = self.x_dim