from abc import ABCMeta, abstractmethod
from typing import Tuple
import torch
from omegaconf import DictConfig
from script.data_handler.data_handler import DataHandler


class GenerationDataHandler(DataHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_input_dim = self.x_dim + self.c_dim + 1
        self.model_condition_dim = self.c_dim
        self.model_output_dim = self.x_dim

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

    def ot_variables(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_x(z0), self.get_x(z1)

    def model_input_terminals(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(z0) == len(z1)
        assert z0.dim() == z1.dim() == 2
        batch_size = len(z0)
        x0 = self.get_x(z0)
        x1 = self.get_x(z1)
        c = self.get_c(z1)
        return torch.cat(
            [x0, c, torch.zeros(batch_size, 1, device=c.device)], dim=1
        ), torch.cat([x1, c, torch.ones(batch_size, 1, device=c.device)], dim=1)

    def expand_model_input(
        self, z: torch.tensor, pre_layer=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, c, t = z.split([self.x_dim, self.c_dim, 1], dim=1)
        return x, c, t

    def convert_target(self, zt: torch.Tensor) -> torch.Tensor:
        return self.get_x(zt)

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
        if t.dim() != 2:
            t = t.reshape(-1, 1)
            t = t.expand(xt.shape[0], 1)
        return torch.cat([xt, targc, t], dim=1)


class ClassiferFreeGuidanceDataHandler(GenerationDataHandler):
    def __init__(
        self,
        cond_drop: float = 0.5,
        lam_max: float = 20,
        lam_min: float = -20,
        c_max: float = 20,
        c_min: float = 0.1,
        v: float = 0.3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.c_dim = self.c_dim + 1  # last dim is for cond drop
        self.model_condition_dim = self.c_dim
        self.model_input_dim = self.x_dim + self.c_dim+1
        self.model_output_dim = self.x_dim
        self.cond_drop = cond_drop
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.c_max = c_max
        self.c_min = c_min
        self.v = v
        self.alpha_sqs = lambda t: torch.clip(
            torch.sqrt(1 / (1 + torch.exp(-t))), 0.1, 1
        )
        self.sigma_sqs = lambda t: torch.sqrt(1 - self.alpha_sqs(t) ** 2)
        self.xi = lambda t: torch.exp(
            -0.25 * (t**2) * (self.c_max - self.c_min) - 0.5 * t * self.c_min
        )
        self.xi_dot = lambda t: (
            -0.5 * t * (self.c_max - self.c_min) - 0.5 * self.c_min
        ) * torch.exp(
            -0.25 * (t**2) * (self.c_max - self.c_min) - 0.5 * t * self.c_min
        )
        self.alpha = lambda t: self.xi(t)
        self.sigma = lambda t: torch.sqrt(1 - self.xi(t) ** 2)
        self.alpha_dot = lambda t: self.xi_dot(t)
        self.sigma_dot = lambda t: -self.xi_dot(t) / self.sigma(t)

    def get_x(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        assert z.shape[1] == self.x_dim + self.c_dim
        return z[:, : self.x_dim]

    def get_c(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        assert z.shape[1] == self.x_dim + self.c_dim
        return z[:, self.x_dim : -1]

    def get_lambda(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        assert z.shape[1] == self.x_dim + self.c_dim
        return z[:, -1, None]

    def reshape_batch(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(x) == len(c)
        assert x.dim() == 2 and c.dim() == 2
        batch_size = len(x)
        lam = torch.multinomial(
            torch.tensor([self.cond_drop, 1 - self.cond_drop], device=c.device),
            num_samples=batch_size,
            replacement=True,
        ).reshape(
            batch_size, 1
        )  # lam = 0 is for drop
        return torch.cat([x, c, lam], dim=1)

    def model_input_terminals(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(z0) == len(z1)
        assert z0.dim() == z1.dim() == 2
        batch_size = len(z0)
        x0 = self.get_x(z0)
        x1 = self.get_x(z1)
        c = self.get_c(z1)
        lam = self.get_lambda(z1)
        c[lam.flatten() == 0] = torch.zeros_like(c[lam.flatten() == 0])
        return torch.cat(
            [
                x0,
                c,
                torch.zeros(batch_size, 1, device=c.device),
                lam.expand(batch_size, 1),
            ],
            dim=1,
        ), torch.cat(
            [
                x1,
                c,
                torch.ones(batch_size, 1, device=c.device),
                lam.expand(batch_size, 1),
            ],
            dim=1,
        )

    def expand_model_input(
        self, z: torch.tensor, pre_layer=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, c, t, lam = z.split([self.x_dim, self.c_dim - 1, 1, 1], dim=1)
        c = torch.cat([c, lam], dim=1)
        return x, c, t

    def inference_model_input(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        initc: torch.Tensor,
        targc: torch.Tensor,
        cond: int = 0,
    ) -> torch.Tensor:
        assert (
            initc.shape == targc.shape
        ), f"shapes of conditions are different {initc.shape}, {targc.shape}"
        assert cond == 0 or cond == 1
        lam = torch.ones(len(targc), 1, device=xt.device) * cond
        if cond == 0:
            targc = torch.zeros_like(targc)
        t = t.reshape(-1, 1).expand(len(xt), 1)
        return torch.cat([xt, targc, t, lam], dim=1)
