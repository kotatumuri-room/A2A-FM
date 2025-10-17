from lightning import LightningModule
from typing import List
from torch.utils.data import Dataset
from omegaconf import DictConfig
from hydra.utils import instantiate
from stochastic_interpolant.multi_marginal_dataset import MultiMarginalDataset
from stochastic_interpolant.utils import KSimplexUniform
import torch


class VelocityField(LightningModule):
    def __init__(self, datasets: List[Dataset], myconfig: DictConfig, k: int):
        super().__init__()
        self.datasets = datasets
        self.K = len(datasets)
        self.k = k
        self.model = instantiate(myconfig.model, _recursive_=False, K=self.K)
        self.config = myconfig
        self.simplex_sampler = KSimplexUniform(self.K)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim_config = {}
        optim_config["optimizer"] = instantiate(
            self.config.optimizer, params=self.model.parameters()
        )
        if hasattr(self.config, "lr_scheduler"):
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate(
                    self.config.lr_scheduler, optimizer=optim_config["optimizer"]
                )
            }
        return optim_config

    def setup(self, stage: str):
        self.multi_dataset = MultiMarginalDataset(self.datasets)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_dataloader(self):
        return instantiate(self.config.train_dataloader, dataset=self.multi_dataset)

    def on_train_start(self):
        self.model.train()

    def training_step(self, batch, batch_idx):
        batch = batch.reshape(-1, self.K, self.datasets.x_dim)
        batch_size = len(batch)
        alphas = self.simplex_sampler.sample(batch_size).to(batch.device)
        xa = torch.einsum("bkd, bk-> bd", (batch, alphas))
        xk = batch[:, self.k, :]
        gk = self(alphas, xa)
        g_sq = torch.einsum("bd, bd->b", (gk, gk))
        xg = torch.einsum("bd, bd->b", (xk, gk))
        loss = torch.mean(g_sq - 2 * xg)
        self.log("train_loss", loss)
        return loss
