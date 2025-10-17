import ot
import torch
from script.data_handler.data_handler import DataHandler


class OptimalTransportCalculator(object):
    """OT calculation class for OTCFM and other models"""

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    @torch.no_grad()
    def dist(self, x, y):
        return ot.dist(x, y, metric="sqeuclidean")

    @torch.no_grad()
    def conduct_sort(self, z0, z1):
        z0_hat, z1_hat = self.data_handler.ot_variables(z0, z1)
        M = self.dist(z0_hat, z1_hat)
        a = torch.ones(M.shape[0])
        b = torch.ones(M.shape[1])
        pi = ot.emd(a, b, M.clone().detach(), numItermax=1000000)
        todat_idx = torch.argmax(pi, axis=1)
        return z0, z1[todat_idx]


class BetaOptimalTransportCalculator(OptimalTransportCalculator):
    """OT calculator for Chemseddine et al. (2024), Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching"""

    def __init__(self, beta: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    @torch.no_grad()
    def dist(self, x, y):
        x_ = self.data_handler.apply_beta(x, self.beta)
        y_ = self.data_handler.apply_beta(y, self.beta)
        return ot.dist(x_, y_)
