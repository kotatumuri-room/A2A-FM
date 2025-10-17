from lightning import LightningModule
import torch
from abc import ABCMeta, abstractmethod
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchdiffeq import odeint
import math
from tqdm.auto import trange, tqdm


class BaseEvaluator(metaclass=ABCMeta):
    def __init__(
        self,
        model: LightningModule,
        myconfig: DictConfig,
        timesteps: int = 2,
        force_steps: int = 100,
    ):
        self.model = model.eval()
        self.config = myconfig
        self.data_handler = instantiate(
            myconfig.data_handler,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        assert timesteps >= 2
        assert force_steps >= 2
        self.timesteps = timesteps
        self.force_steps = force_steps

    @torch.no_grad()
    def integrate_ct(
        self,
        initx: torch.Tensor,
        targc: torch.Tensor,
        initc=None,
        boosting: float = 1.0,
    ) -> torch.Tensor:
        if initc is None:
            initc = targc

        ts = torch.linspace(0, 1, self.timesteps, device=initx.device)
        traj = [
            initx.clone(),
        ]
        for i in trange(len(ts) - 1, leave=False):
            c0 = initc * (1 - ts[i]) + targc * ts[i]
            c1 = initc * (1 - ts[i + 1]) + targc * ts[i + 1]

            def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
                z_hat = self.data_handler.inference_model_input(t, xt, c0, c1)
                return self.model(z_hat) * boosting

            partial_ts = torch.linspace(0, 1, self.force_steps, device=initx.device)
            partial_traj = odeint(vel, traj[-1], partial_ts, method="dopri5")
            traj.append(partial_traj[-1])
        traj = torch.stack(traj, dim=0)
        return traj

    @torch.no_grad()
    def integrate(
        self,
        initx: torch.Tensor,
        targc: torch.Tensor,
        initc=None,
        boosting: float = 1.0,
    ) -> torch.Tensor:
        if initc is None:
            initc = targc

        ts = torch.linspace(0, 1, self.timesteps, device=initx.device)

        def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            z_hat = self.data_handler.inference_model_input(t, xt, initc, targc)
            return self.model(z_hat) * boosting

        traj = odeint(vel, initx, ts, method="dopri5")
        return traj

    @torch.no_grad()
    def cfg_flow_integrate(
        self, initx: torch.Tensor, targc: torch.Tensor, w: float = 1.0
    ) -> torch.Tensor:
        initc = targc
        ts = torch.linspace(1, 0, self.timesteps, device=initx.device)

        def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            z_hat_nocond = self.data_handler.inference_model_input(
                t, xt, initc, targc, cond=0
            )
            z_hat_cond = self.data_handler.inference_model_input(
                t, xt, initc, targc, cond=1
            )
            return self.model(z_hat_nocond) * (1 - w) + self.model(z_hat_cond) * w

        traj = odeint(vel, initx, ts, method="dopri5")
        return traj
    @torch.no_grad()
    def partial_diffusion(
        self,
        initx: torch.Tensor,
        targc: torch.Tensor,
        timesteps: float = 0.5,
        w: float = 0.3,
    ):
        assert 0 <= timesteps <= 1
        ts = torch.linspace(timesteps, 0, int(self.timesteps*timesteps), device=initx.device)
        b = math.atan(math.exp(-self.data_handler.lam_max * 0.5))
        a = math.atan(math.exp(-self.data_handler.lam_min * 0.5)) - b
        lambdas = -2 * torch.log(torch.tan(a * ts + b))

        def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            z_hat_nocond = self.data_handler.inference_model_input(
                t, xt, targc, targc, cond=0
            )
            z_hat_cond = self.data_handler.inference_model_input(
                t, xt, targc, targc, cond=1
            )
            return -self.model(z_hat_nocond) * w + self.model(z_hat_cond) * (1 + w)

        traj = []
        for u in torch.flip(lambdas,(0,)):
            xt = initx * self.data_handler.alpha_sqs(u) + self.data_handler.sigma_sqs(
                u
            ) * torch.randn_like(initx)
            traj.append(xt)
        for i, (lam, t) in tqdm(
            enumerate(zip(lambdas, ts)), desc="generating", total=len(ts)
        ):
            zt = traj[-1]
            eps = vel(t, zt)
            xt = (
                zt - self.data_handler.sigma_sqs(lam) * eps
            ) / self.data_handler.alpha_sqs(lam)
            if i < len(ts) - 1:
                tilde_sigma_lam_lam = torch.sqrt(
                    (1 - torch.exp(lam - lambdas[i + 1]))
                ) * self.data_handler.sigma_sqs(lambdas[i + 1])
                sigma_lam_lam = torch.sqrt(
                    (1 - torch.exp(lam - lambdas[i + 1]))
                ) * self.data_handler.sigma_sqs(lam)
                sigma_sq = ((sigma_lam_lam) ** (2 * self.data_handler.v)) * (
                    (tilde_sigma_lam_lam) ** (2 * (1 - self.data_handler.v))
                )
                alpha = self.data_handler.alpha_sqs(lam)
                alpha_prime = self.data_handler.alpha_sqs(lambdas[i + 1])
                mu = torch.exp(lam - lambdas[i + 1]) * (alpha_prime / alpha) * zt + (
                    (1 - torch.exp(lam - lambdas[i + 1])) * alpha_prime * xt
                )
                traj.append(mu + torch.sqrt(sigma_sq) * torch.randn_like(mu))
            else:
                traj.append(xt)
        return torch.stack(traj, dim=0)

    @torch.no_grad()
    def cfg_flow_nocond_integrate(
        self, initx: torch.Tensor, initc: torch.Tensor
    ) -> torch.Tensor:
        ts = torch.linspace(1, 0, self.timesteps, device=initx.device)

        def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            z_hat_nocond = self.data_handler.inference_model_input(
                t, xt, initc, initc, cond=0
            )
            return self.model(z_hat_nocond)

        traj = odeint(vel, initx, ts, method="dopri5")
        return traj

    @torch.no_grad()
    def cfg_diffusion_integrate(
        self, initx: torch.Tensor, targc: torch.Tensor, w: float = 1.0
    ):
        initc = targc
        ts = torch.linspace(1, 0, self.timesteps, device=initx.device)
        b = math.atan(math.exp(-self.data_handler.lam_max * 0.5))
        a = math.atan(math.exp(-self.data_handler.lam_min * 0.5)) - b
        lambdas = -2 * torch.log(torch.tan(a * ts + b))
        def vel(t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
            z_hat_nocond = self.data_handler.inference_model_input(
                t, xt, initc, targc, cond=0
            )
            z_hat_cond = self.data_handler.inference_model_input(
                t, xt, initc, targc, cond=1
            )
            return -self.model(z_hat_nocond) * w + self.model(z_hat_cond) * (1 + w)

        traj = [initx]
        for i, (lam, t) in tqdm(
            enumerate(zip(lambdas, ts)), desc="generating", total=len(ts)
        ):
            zt = traj[-1]
            eps = vel(t, zt)
            xt = (
                zt - self.data_handler.sigma_sqs(lam) * eps
            ) / self.data_handler.alpha_sqs(lam)
            if i < len(ts) - 1:
                tilde_sigma_lam_lam = torch.sqrt(
                    (1 - torch.exp(lam - lambdas[i + 1]))
                ) * self.data_handler.sigma_sqs(lambdas[i + 1])
                sigma_lam_lam = torch.sqrt(
                    (1 - torch.exp(lam - lambdas[i + 1]))
                ) * self.data_handler.sigma_sqs(lam)
                sigma_sq = ((sigma_lam_lam) ** (2 * self.data_handler.v)) * (
                    (tilde_sigma_lam_lam) ** (2 * (1 - self.data_handler.v))
                )
                alpha = self.data_handler.alpha_sqs(lam)
                alpha_prime = self.data_handler.alpha_sqs(lambdas[i + 1])
                mu = torch.exp(lam - lambdas[i + 1]) * (alpha_prime / alpha) * zt + (
                    (1 - torch.exp(lam - lambdas[i + 1])) * alpha_prime * xt
                )
                traj.append(mu + torch.sqrt(sigma_sq) * torch.randn_like(mu))
            else:
                traj.append(xt)
        return torch.stack(traj, dim=0)

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def validation(self, batch: torch.Tensor) -> dict:
        pass
