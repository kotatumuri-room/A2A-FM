from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
import pickle


CLASS_NAMES = ["Bangs", "Eyeglasses", "No_Beard", "Smiling", "Young"]


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        preprocessed_root: str,
        x_name: str,
        c_name: str,
        c_data: np.ndarray,
        i: int,
        c_indices:list=[0,1,2,3,4],
        upto: int = None,
        size:int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        self.file_list = np.asarray(file_list)
        self.size = size if size is not None else len(c_data)
        self.x_name = x_name
        self.c_name = c_name
        self.c_indices = c_indices
        self.cdata = c_data
        self.file_list = self.file_list[self.cdata == i]
        self.cdata = self.cdata[self.cdata == i]
        if upto is not None:
            self.file_list = self.file_list[:upto]
            self.cdata = self.cdata[:upto]
        self.x_dim = 12288
        self.c_dim = 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        file = self.file_list[idx%len(self.file_list)]
        data = np.load(file)
        x = data[self.x_name]
        c = self.cdata[idx%len(self.file_list)]
        assert x.flatten().shape[0] == self.x_dim
        x = x.reshape(self.x_dim)
        c = c.reshape(self.c_dim)
        return x, c


class MultiMarginalCelebA(Dataset):

    def __init__(
        self,
        K: int,
        preprocessed_root: str,
        k_means_dir: str,
        x_name: str,
        c_name: str,
        c_indices:list=[0,1,2,3,4],
        upto: int = None,
    ):
        self.datas = []
        file_list = list(Path(preprocessed_root).rglob("*.npz"))
        self.file_list = np.asarray(file_list)
        self.x_name = x_name
        self.c_name = c_name
        self.c_indices = c_indices
        self.x_dim = 12288
        self.c_dim = 1
        if Path(k_means_dir).exists():
            kmeans_model = pickle.load(open(k_means_dir, "rb"))
            c = self.load_conditions()
            labels = kmeans_model.predict(c)
        else:
            c = self.load_conditions()
            kmeans_model = KMeans(n_clusters=K, random_state=0, verbose=True).fit(c)
            labels = kmeans_model.labels_
            Path(k_means_dir).parent.mkdir(exist_ok=True, parents=True)
            pickle.dump(kmeans_model, open(k_means_dir, "wb"))
        self.kmeans_model = kmeans_model
        self.datasets = [
            CelebAHQDataset(preprocessed_root, x_name, c_name, labels, i,c_indices, upto)
            for i in range(K)
        ]
        self.K = K

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
        return self.datasets[idx]

    def load_conditions(self):
        cs = []
        for file in tqdm(self.file_list, desc="reading files for kmeans"):
            data = np.load(file)
            c = data[self.c_name][self.c_indices]
            cs.append(c)
        return np.stack(cs, axis=0)
