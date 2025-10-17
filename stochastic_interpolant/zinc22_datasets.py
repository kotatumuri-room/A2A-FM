from torch.utils.data import Dataset
import numpy as np
from pathlib import Path, PosixPath
from typing import List
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import pickle


class ZINC22_Dataset(Dataset):
    def __init__(
        self,
        preprocessed_root: str,
        k_means_model: KMeans,
        i: int,
        x_name: str = "emb_smiles",
        cond_columns: List[str] = ["cdfd_logp", "cdfd_qed", "cdfd_tpsa", "cdfd_SA"],
        upto: int = None,
        use_scs: bool = True,
        **kwargs,
    ):
        super().__init__()
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        np.random.shuffle(file_list)
        self.file_list = np.array(file_list)
        if upto is not None:
            self.file_list = self.file_list[:upto]
        self.cond_names = ["embedded_" + c[5:] for c in cond_columns]
        self.cond_columns = cond_columns
        self.x_name = x_name
        self.k_means_model = k_means_model
        self.i = i

    def __len__(self):
        return len(self.file_list)

    def screen_kmeans(self, x: np.ndarray, c: np.ndarray, cdfd: np.ndarray):
        labels = self.k_means_model.predict(cdfd)
        c = c[labels == self.i]
        x = x[labels == self.i]
        if len(x) == 0:
            return False, False
        return x, c

    def __getitem__(self, idx):
        x, c = False, False
        i = 100
        n = 0
        while x is False and n < 100:
            x, c, cdfd = self.read_point(self.file_list[i])
            x, c = self.screen_kmeans(x, c, cdfd)
            i = np.random.randint(len(self.file_list))
            n += 1
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
        cdfd = []
        for url in urls_:
            with open(url, "rb") as fp:
                try:
                    data = np.load(fp)
                    datas.append(data[self.x_name])
                    labels.append(
                        np.concatenate(
                            [data[c_name] for c_name in self.cond_names], axis=1
                        )
                    )
                    cdfd.append(
                        np.stack(
                            [data[c_name] for c_name in self.cond_columns], axis=-1
                        )
                    )
                except:
                    urls_.append(self.file_list[np.random.randint(len(self.file_list))])
        return (
            np.concatenate(datas, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(cdfd, axis=0),
        )


class MultiMarginalZinc22(Dataset):

    def __init__(
        self,
        K: int,
        preprocessed_root: str,
        k_means_dir: str,
        x_name: str = "emb_smiles",
        cond_columns: List[str] = ["cdfd_logp", "cdfd_qed", "cdfd_tpsa", "cdfd_SA"],
        upto: int = None,
        use_scs: bool = True,
    ):
        self.datas = []
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        self.file_list = np.asarray(file_list)
        np.random.shuffle(self.file_list)
        self.x_name = x_name
        self.x_dim = 512
        self.c_dim = len(cond_columns)
        self.cond_names = ["embedded_" + c[5:] for c in cond_columns]
        self.cond_columns = cond_columns
        if Path(k_means_dir).exists():
            kmeans_model = pickle.load(open(k_means_dir, "rb"))
        else:
            c = self.read_point(self.file_list[:30])
            kmeans_model = KMeans(n_clusters=K, random_state=0, verbose=True).fit(c)
            Path(k_means_dir).parent.mkdir(exist_ok=True, parents=True)
            pickle.dump(kmeans_model, open(k_means_dir, "wb"))
        self.datasets = [
            ZINC22_Dataset(
                preprocessed_root, kmeans_model, i, x_name, cond_columns, upto, use_scs
            )
            for i in range(K)
        ]
        self.kmeans_model=kmeans_model
        self.K = K

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
        return self.datasets[idx]

    def read_point(self, urls: List[str] | str | PosixPath):
        if isinstance(urls, str) or isinstance(urls, PosixPath):
            urls_ = [
                urls,
            ]
        else:
            urls_ = list(urls)
        labels = []
        for url in urls_:
            with open(url, "rb") as fp:
                try:
                    data = np.load(fp)
                    labels.append(
                        np.stack([data[c_name] for c_name in self.cond_columns], axis=1)
                    )
                except:
                    urls_.append(self.file_list[np.random.randint(len(self.file_list))])
        return np.concatenate(labels, axis=0)
