import torch
from script.process_generator.base_process import ProcessGenerator
from hydra.utils import instantiate
import math

NOISE_SCHEDULE = {"linear": lambda t: t}


class DDPMDiffusion(ProcessGenerator):
    """
    Implementation for Ho et al. (2020) Denoising Diffusion Probabilistic Models.
    """

    def __init__(self, timesteps: int = 1000, noise_schedule="linear"):
        self.timesteps = timesteps
        ts = torch.arange(0, self.timesteps + 1)
        self.betas = NOISE_SCHEDULE[noise_schedule](ts)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas)

    def descritize_t(self, t):
        return torch.round(t * self.timesteps).to(torch.int32)

    def set_batch(self, z0: torch.Tensor, z1: torch.Tensor):
        self.z0, self.z1 = z0, z1

    def construct_target(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        self.noise = torch.randn_like(self.z0)
        return self.noise

    def get_model_input(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        z0_hat, z1_hat = self.data_handler.model_input_terminals(self.z0, self.z1)
        t = self.descritize_t(t)
        alpha_bars = self.alpha_bars[t]
        zt = torch.sqrt(alpha_bars) * self.z0 + torch.sqrt(1 - alpha_bars) * self.noise
        zt[:, self.data_handler.x_dim :] = z0_hat[:, self.data_handler.x_dim :]
        zt[:, self.data_handler.x_dim + self.data_handler.c_dim] = t.squeeze(1)
        return zt

    def sample_t(self) -> torch.Tensor:
        batch_size = len(self.z0)
        return torch.rand(batch_size, 1, device=self.z0.device)


class CFGDiffusion(ProcessGenerator):
    """
    Implementation for Ho et al. (2022) Classifer-free diffusion guidance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_batch(self, z0: torch.Tensor, z1: torch.Tensor):
        self.z0, self.z1 = z0, z1

    def construct_target(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        return self.data_handler.convert_target(self.z0)

    def get_model_input(self, t: torch.Tensor) -> torch.Tensor:
        assert len(self.z0) == len(self.z1) == len(t)
        z0_hat, z1_hat = self.data_handler.model_input_terminals(self.z0, self.z1)
        b = math.atan(math.exp(-self.data_handler.lam_max * 0.5))
        a = math.atan(math.exp(-self.data_handler.lam_min * 0.5)) - b
        u = -2 * torch.log(torch.tan(a * t + b))
        zt = (
            self.data_handler.alpha_sqs(u) * z1_hat
            + self.data_handler.sigma_sqs(u) * z0_hat
        )
        zt[:, self.data_handler.x_dim :] = z0_hat[:, self.data_handler.x_dim :]
        zt[:, self.data_handler.x_dim + self.data_handler.c_dim - 1] = t.squeeze(1)
        return zt

    def sample_t(self) -> torch.Tensor:
        batch_size = len(self.z0)
        u = torch.rand(batch_size, 1, device=self.z0.device)
        return u
