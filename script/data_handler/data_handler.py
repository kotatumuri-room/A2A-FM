from abc import ABCMeta, abstractmethod
from typing import Tuple
import torch
from omegaconf import DictConfig


class DataHandler(metaclass=ABCMeta):
    def __init__(self, data_config: DictConfig):
        self.config = data_config
        self.x_dim = data_config.x_dim
        self.c_dim = data_config.c_dim
        self.model_input_dim = None
        self.model_output_dim = None

    @abstractmethod
    def get_x(self, z: torch.Tensor) -> torch.Tensor:
        """return x from z"""
        pass

    @abstractmethod
    def get_c(self, z: torch.Tensor) -> torch.Tensor:
        """return c from z"""
        pass

    @abstractmethod
    def reshape_batch(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """return z1 from x and c of dataset"""
        pass

    def ot_variables(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """return ot input"""
        raise NotImplementedError

    def apply_beta(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def convert_target(self, zt: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def model_input_terminals(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def expand_model_input(
        self, z: torch.tensor, pre_layer=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def split_for_symmetric(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inference_model_input(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        initc: torch.Tensor,
        targc: torch.Tensor,
    ) -> torch.Tensor:
        pass
