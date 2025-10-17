import numpy as np
from script.datasets.base_dataset import BaseDataset
from pathlib import Path
from typing import List

CLASS_NAMES = ["Bangs", "Eyeglasses", "No_Beard", "Smiling", "Young"]


class CelebAHQDataset(BaseDataset):
    def __init__(
        self,
        preprocessed_root: str,
        x_name: str,
        c_name: str,
        c_indices: List[int] = [0, 1, 2, 3, 4],
        upto: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        self.file_list = np.asarray(file_list)
        if upto is not None:
            self.file_list = self.file_list[:upto]
        self.x_name = x_name
        self.c_name = c_name
        self.c_indices = c_indices

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        with open(file,"rb") as fp:
            data = np.load(fp)
            x = data[self.x_name]
            c = data[self.c_name][self.c_indices]
            assert c.shape[-1] == self.c_dim, f"c:{c.shape}, cdim {self.c_dim}"
            assert x.flatten().shape[0] == self.x_dim
            x = x.reshape(self.x_dim)
            c = c.reshape(self.c_dim)
        return x, c


class CelebAHQBinaryDataset(CelebAHQDataset):
    def __getitem__(self, idx):
        file = self.file_list[idx]
        with open(file,"rb") as fp:
            data = np.load(fp)
            x = data[self.x_name]
            c = data[self.c_name][self.c_indices]
            assert c.shape[-1] == self.c_dim, f"c:{c.shape}, cdim {self.c_dim}"
            assert x.flatten().shape[0] == self.x_dim
            x = x.reshape(self.x_dim)
            c = c.reshape(self.c_dim)
            c = c > 0
        return x, c.astype(int)