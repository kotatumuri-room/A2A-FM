from lightning import LightningModule
import torch
from script.evaluator.base_evaluator import BaseEvaluator
from typing import List
from matplotlib import pyplot as plt
from lightning.pytorch import loggers as pl_loggers
import numpy as np


class SyntheticEvaluator(BaseEvaluator):
    def __init__(self, eval_clist: List, **kwargs):
        super().__init__(**kwargs)
        self.eval_clist = torch.tensor(eval_clist)

    def evaluate(self):
        raise NotImplementedError

    def validation(self, batch: torch.Tensor) -> dict:
        x, c, _, _ = batch
        for targc in self.eval_clist:
            traj = self.integrate(x, targc.unsqueeze(0).expand_as(c).to(c.device), c)
            fig = self.plot_traj(traj.cpu().numpy())
            fig_array = self.plt_to_npy(fig)
            plt.close()
            self.log_image(fig_array, f"transfer to {list(targc.cpu().numpy())}")

    def log_image(self, image, logname):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.model.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        tb_logger.add_image(logname, image, self.model.global_step)

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
        return fig

    def plt_to_npy(self, fig):
        fig.canvas.draw() 
        plot_image = fig.canvas.renderer._renderer 
        plot_image_array = np.array(plot_image).transpose(2, 0, 1)
        return plot_image_array
