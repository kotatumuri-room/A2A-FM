from torch.distributions.dirichlet import Dirichlet
import torch
from typing import List
from lightning import LightningModule



class KSimplexUniform(object):
    def __init__(self, k: int):
        self.k = k
        self.sampler = Dirichlet(torch.ones(k))

    def sample(self, n: int):
        return self.sampler.sample((n,))


class Velocity(torch.nn.Module):
    def __init__(self, gs: List[LightningModule], alpha: LightningModule, reverse:bool =False):
        super().__init__()
        self.gs = gs
        self.alpha = alpha
        self.reverse = reverse
    @torch.no_grad()
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        if self.reverse:
            t = 1 - t
        if t.dim() == 0:
            t = t.reshape(-1).expand(len(x))
        alpha = self.alpha(t)
        alpha_dot = self.alpha.model.dot(t)
        gs = torch.stack([gk(alpha, x) for gk in self.gs], dim=1)
        v = torch.einsum("bk,bkd->bd", (alpha_dot, gs))
        if self.reverse:
            return -1*v
        return v 


