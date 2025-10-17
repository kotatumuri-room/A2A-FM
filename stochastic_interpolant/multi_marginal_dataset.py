from torch.utils.data import Dataset
from typing import List
import torch


class MultiMarginalDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.K = len(self.datasets)
        self.num_samples = min([len(dataset) for dataset in datasets])
        assert self.num_samples > 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_list = []
        max_size = 0
        for k in range(self.K):
            x, _ = self.datasets[k][idx]
            d = self.datasets.x_dim
            x_list.append(torch.tensor(x).reshape(-1, d))
            max_size = max(len(x_list[-1]), max_size)
        out = torch.zeros(max_size, self.K, d)
        for k in range(self.K):
            out[: len(x_list[k]), k, :] = x_list[k]
            if max_size - len(x_list[k]) > 0:
                out[len(x_list[k]) :, k, :] = x_list[k][
                    torch.multinomial(
                        torch.ones(len(x_list[k])),
                        max_size - len(x_list[k]),
                        replacement=True,
                    )
                ]
        return out
