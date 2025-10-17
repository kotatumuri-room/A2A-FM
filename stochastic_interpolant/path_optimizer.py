from lightning import LightningModule
from typing import List
from torch.utils.data import Dataset
from omegaconf import DictConfig
from hydra.utils import instantiate
from stochastic_interpolant.multi_marginal_dataset import MultiMarginalDataset
import torch


class PathOptimizer(LightningModule):
    def __init__(
        self,
        gs: List[LightningModule],
        datasets: List[Dataset],
        myconfig: DictConfig,
        i: int,
        j: int,
    ):
        super().__init__()
        self.gs = gs
        self.datasets = datasets
        self.K = len(datasets)
        self.i = i
        self.j = j
        self.model = instantiate(myconfig.model, _recursive_=False, K=self.K, i=i, j=j)
        self.config = myconfig

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
        batch = batch.reshape(-1,self.K,self.datasets.x_dim)
        batch_size = len(batch)
        self.gs = [g.to(batch.device) for g in self.gs]
        t = torch.rand(batch_size,requires_grad=True, device=batch.device)
        alphas = self(t)
        alphas_dot = self.model.dot(t)  # bk
        xa = torch.einsum("bkd, bk-> bd", (batch, alphas))  # bd
        gks = torch.stack([g(alphas, xa) for g in self.gs], dim=1)  # bkd
        vel = torch.einsum("bkd, bk->bd", (gks, alphas_dot))
        cost = torch.einsum("bd, bd->b", (vel, vel))
        loss = torch.mean(cost)
        self.log("train_loss", loss)
        return loss
