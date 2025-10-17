import torch
from abc import abstractmethod, ABCMeta
from script.data_handler.data_handler import DataHandler


class ProcessGenerator(metaclass=ABCMeta):
    """process generator for training and inference"""

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        self.z1 = None
        self.z0 = None

    @abstractmethod
    def set_batch(self,z0:torch.Tensor,z1:torch.Tensor)->None:
        """set batch and prepare z0"""
        

    @abstractmethod
    def construct_target(self, t: torch.Tensor) -> torch.Tensor:
        """
        creates target vector field or score function
        z0: (b d), z1: (b d), t: (b 1)
        returns: (b d)
        """

    @abstractmethod
    def get_model_input(self, t: torch.Tensor) -> torch.Tensor:
        """
        creates model input for vector field or score function
        z0: (b d), z1: (b d)
        returns: (b dim_in)
        """

    @abstractmethod
    def sample_t(self) -> torch.Tensor:
        """
        returns samples of time in (0,1)
        """
