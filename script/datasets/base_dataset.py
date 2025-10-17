from torch.utils.data import Dataset
from numpy import ndarray
from omegaconf import DictConfig
import numpy as np


class BaseDataset(Dataset):
    """Base class for STFM"""

    def __init__(self, data_config: DictConfig):
        self.x_dim: int = data_config.x_dim
        self.c_dim: int = data_config.c_dim
        self.xdata: ndarray = None
        self.cdata: ndarray = None

    def __len__(self) -> int:
        return len(self.xdata)

    def __getitem__(self, idx: int):
        return self.xdata[idx], self.cdata[idx]


class NormalGaussian(BaseDataset):
    """Base class for STFM"""

    def __init__(self, data_config: DictConfig):
        self.x_dim: int = data_config.x_dim
        self.c_dim: int = data_config.c_dim

    def __len__(self) -> int:
        return 10000000000000000

    def __getitem__(self, idx: int):
        return (
            np.random.randn((self.x_dim)).astype(np.float32),
            np.random.randn((self.c_dim)).astype(np.float32),
        )


class DoubleTerminalDataset(BaseDataset):
    def __init__(
        self, data_config: DictConfig, dataset1: BaseDataset, dataset2: BaseDataset
    ):
        assert dataset1.x_dim == dataset2.x_dim
        assert dataset1.c_dim == dataset2.c_dim
        self.x_dim = dataset1.x_dim
        self.c_dim = dataset2.c_dim
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        length = 0
        if isinstance(dataset1, NormalGaussian):
            length = len(dataset2)
        if isinstance(dataset2, NormalGaussian):
            length = len(dataset1)
        else:
            length = min(len(dataset1), len(dataset2))
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        idx1 = np.random.choice(len(self), 1, replace=True)[0]
        return self.dataset1[idx], self.dataset2[idx1]
