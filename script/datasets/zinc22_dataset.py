from typing import List
import numpy as np
from script.datasets.base_dataset import BaseDataset
from pathlib import Path, PosixPath


class ZINC22_Dataset(BaseDataset):
    def __init__(
        self,
        preprocessed_root: str,
        x_name: str = "emb_smiles",
        cond_columns: List[str] = ["cdfd_logp", "cdfd_qed", "cdfd_tpsa", "cdfd_SA"],
        upto: int = None,
        use_scs: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        np.random.shuffle(file_list)
        self.file_list = np.array(file_list)
        if upto is not None:
            self.file_list = self.file_list[:upto]
        self.cond_names = ["embedded_" + c[5:] for c in cond_columns]
        self.cond_columns = cond_columns
        self.x_name = x_name

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        x, c = self.read_point(self.file_list[idx])
        return x, c

    def read_point(self, urls: List[str] | str | PosixPath):
        if isinstance(urls, str) or isinstance(urls, PosixPath):
            urls_ = [
                urls,
            ]
        else:
            urls_ = urls
        labels = []
        datas = []
        for url in urls_:
            with open(url,"rb") as fp:
                try:
                    data = np.load(fp)
                    datas.append(data[self.x_name])
                    labels.append(
                        np.concatenate([data[c_name] for c_name in self.cond_names], axis=1)
                    )
                except :
                    urls_.append(self.file_list[np.random.randint(len(self.file_list))])
        return np.concatenate(datas, axis=0), np.concatenate(labels, axis=0)
