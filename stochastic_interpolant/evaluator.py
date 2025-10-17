from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from omegaconf import DictConfig
from stochastic_interpolant.multi_marginal_dataset import MultiMarginalDataset
from hydra.utils import instantiate
from tqdm.auto import tqdm
from torch.nn.functional import mse_loss
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from stochastic_interpolant.utils import Velocity


class PathEvaluator(object):
    def __init__(
        self,
        dataset0: Dataset,
        dataset1: Dataset,
        vel: Velocity,
        config: DictConfig,
        log_dir: str,
        c0,
        c1,
    ):
        dataset = MultiMarginalDataset([dataset0, dataset1])
        self.vel = vel
        self.device = next(vel.parameters().__iter__()).device
        self.dataloader = instantiate(config.val_dataloader, dataset=dataset)
        self.timesteps = config.timesteps
        self.logger = SummaryWriter(logdir=log_dir, comment="hohe")
        self.c0 = c0
        self.c1 = c1

    @torch.no_grad()
    def validate(self):
        for batch in tqdm(self.dataloader, desc="evaluating"):
            x0, x1 = torch.split(batch.to(self.device), [1, 1], dim=1)
            x0 = x0.flatten(1)
            x1 = x1.flatten(1)
            ts = torch.linspace(0, 1, self.timesteps, device=self.device)
            traj = odeint(self.vel, x0, ts, method="dopri5")
            predx = traj[-1]
            mse = mse_loss(predx, x1)
            self.logger.add_scalar(f"xloss {self.c0} to {self.c1}", mse)
            fig = self.plot_traj(traj.cpu().numpy())
            fig_array = self.plt_to_npy(fig)
            self.logger.add_image(f"trajectory {self.c0} to {self.c1}", fig_array)

    def plot_traj(self, traj: torch.Tensor) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.scatter(traj[:, :, 0], traj[:, :, 1], c="tan", s=0.5, alpha=0.1)
        ax.scatter(
            traj[-1, :, 0],
            traj[-1, :, 1],
            c="cyan",
            s=1,
            alpha=1,
            label="generated points",
        )
        ax.scatter(
            traj[0, :, 0],
            traj[0, :, 1],
            c="magenta",
            s=1,
            alpha=1,
            label="initial points",
        )
        ax.legend(fontsize=18)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        return fig

    def plt_to_npy(self, fig):
        fig.canvas.draw()  
        plot_image = fig.canvas.renderer._renderer  
        plot_image_array = np.array(plot_image).transpose(2, 0, 1)
        return plot_image_array


