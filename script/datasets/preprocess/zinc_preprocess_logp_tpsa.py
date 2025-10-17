import torch
import pickle
import gzip
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Descriptors import TPSA
from rdkit.Chem.Crippen import MolLogP
from coatiLDM.data.transforms import cdf, safe_embed_scalar
from coatiLDM.models.coati.io import load_coati2
import itertools
from tqdm.auto import tqdm
import numpy as np
from coatiLDM.common.utils import batch_iterable
import os


def splitparN(iterable, N=3):
    for i, item in itertools.groupby(enumerate(iterable), lambda x: x[0] // N):
        yield (x[1] for x in item)


class ZINC_Preprocessor(object):
    def __init__(
        self,
        data_root: str,
        save_root: str,
        COATI2_DOCS: dict,
        device="cuda",
        batch_size: int = 1024,
        c_embed_dim: int = 32,
        cdf_path=None,
    ):
        data_root = Path(data_root)
        gz_files = list(data_root.rglob("*.gz"))
        model_doc = COATI2_DOCS["general_doc"]
        self.encoder, self.tokenizer = load_coati2(
            model_doc, freeze=True, device=device
        )
        self.device = device
        self.c_embed_dim = c_embed_dim
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        np.random.shuffle(gz_files)
        gz_files = gz_files
        if cdf_path is not None:
            cdf = pickle.load(open(cdf_path, "rb"))
            self.cdf_tpsa = cdf["cdf_tpsa"]
            self.cdf_logp = cdf["cdf_logp"]
        else:
            self.cdf_tpsa = self.construct_cdf_tpsa(gz_files, num_samples=10000)
            self.cdf_logp = self.construct_cdf_logp(gz_files, num_samples=10000)
            with open(self.save_root / "cond_cdfs.pkl", "wb") as f:
                pickle.dump({"cdf_tpsa": self.cdf_tpsa, "cdf_logp": self.cdf_logp}, f)

        for i, files in tqdm(
            enumerate(splitparN(gz_files, N=10)),
            desc="preprocessing",
            total=len(gz_files) // 10,
        ):  # 10個ずつ読む
            smile_batch = []
            for file in tqdm(files, desc="loading files", leave=False, total=10):
                smiles_list = self.read_gz(
                    file, threshold=1e5, verbose=False, chunk_size=1000
                )
                smile_batch += smiles_list
            np.random.shuffle(smile_batch)
            for j, smile in enumerate(batch_iterable(smile_batch, n=batch_size)):
                data = self.smiles_to_training_data(smile)
                self.save_data(data, i=i, j=j)
                data = self.smiles_to_coati_data(smile)
                self.save_coati(data)

    def save_coati(self, data: list):
        coati_path = self.save_root / f"coati_data.pkl"
        if coati_path.exists():
            with open(coati_path, "rb") as f:
                d = pickle.load(f)
            d = d + data
        else:
            d = data
        with open(coati_path, "wb") as f:
            pickle.dump(d, f)

    def save_data(self, data: dict, i: int, j: int):
        save_path = self.save_root / f"batch_{i}_{j}"
        if save_path.exists():
            raise RuntimeWarning(f"file {save_path} exists we are overwriting the file")
        np.savez(save_path, **data)

    def construct_cdf_tpsa(
        self, files_list, num_samples=10, min_size=10, chunk_size=10
    ):
        sampled_files = np.random.choice(
            np.array(files_list, dtype=str), size=num_samples, replace=True
        )
        qed_samples = []
        for file in tqdm(sampled_files):
            smiles_list = self.read_gz(file, threshold=min_size, chunk_size=chunk_size)
            if len(smiles_list) < min_size:
                continue
            smiles = np.random.choice(smiles_list, size=min_size)
            qed_samples = qed_samples + [
                (
                    self.get_tpsa(Chem.MolFromSmiles(smi))
                    if Chem.MolFromSmiles(smi) is not None
                    else 0.0
                )
                for smi in smiles
            ]

        print(
            "max:",
            np.array(qed_samples).max(),
            "min:",
            np.array(qed_samples).min(),
        )
        return cdf(
            np.array(qed_samples),
            min_bin=0,
            max_bin=np.array(qed_samples).max(),
            verbose=True,
        )

    def construct_cdf_logp(
        self, files_list, num_samples=10, min_size=10, chunk_size=10
    ):
        sampled_files = np.random.choice(
            np.array(files_list, dtype=str), size=num_samples, replace=True
        )
        qed_samples = []
        for file in tqdm(sampled_files):
            smiles_list = self.read_gz(file, threshold=min_size, chunk_size=chunk_size)
            if len(smiles_list) < min_size:
                continue
            smiles = np.random.choice(smiles_list, size=min_size)
            qed_samples = qed_samples + [
                (
                    self.get_logp(Chem.MolFromSmiles(smi))
                    if Chem.MolFromSmiles(smi) is not None
                    else 0.0
                )
                for smi in smiles
            ]
        print(
            "max:",
            np.array(qed_samples).max(),
            "min:",
            np.array(qed_samples).min(),
        )
        return cdf(
            np.array(qed_samples),
            min_bin=np.array(qed_samples).min(),
            max_bin=np.array(qed_samples).max(),
            verbose=True,
        )

    def get_qed(self, mol):
        try:
            return QED.qed(mol)
        except:
            return 0.0

    def get_logp(self, mol):
        try:
            return MolLogP(mol)
        except:
            return 0.0

    def get_tpsa(self, mol):
        try:
            return TPSA(mol)
        except:
            return 0.0

    def screen_smiles(self, smiles):
        smiles = np.array(
            [
                smi
                for smi in tqdm(smiles, desc="screening smiles", leave=False)
                if Chem.MolFromSmiles(smi) is not None
            ]
        )
        return np.array(smiles)

    def read_gz(
        self,
        filename,
        chunk_size=10000,
        threshold=1e10,
        verbose=False,
    ):
        output = []
        pb = tqdm(
            desc=f"reading:{filename} size:{os.path.getsize(filename)}",
            leave=False,
            total=threshold,
        )
        with gzip.open(filename, "rt") as f:
            for _ in range(int(threshold // chunk_size)):
                chunk = []
                for _ in range(chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    smiles = line.split("\t")
                    chunk.append(smiles[0])
                pb.update(len(chunk))
                if not chunk:
                    break
                chunk = list(self.screen_smiles(chunk))
                output = output + chunk
                pb.set_description(
                    f"reading:{filename} size:{os.path.getsize(filename)} current_size: {len(output)}"
                )
        return output

    def smiles_to_training_data(self, smiles):
        toks, _ = self.tokenizer.batch_smiles(smiles)
        embs = (
            self.encoder.encode_tokens(toks.to(self.device), self.tokenizer)
            .cpu()
            .numpy()
        )
        tpsas = [
            (
                self.get_tpsa(Chem.MolFromSmiles(smi))
                if Chem.MolFromSmiles(smi) is not None
                else -1
            )
            for smi in smiles
        ]
        logps = [
            (
                self.get_logp(Chem.MolFromSmiles(smi))
                if Chem.MolFromSmiles(smi) is not None
                else None
            )
            for smi in smiles
        ]
        cdfd_tpsa = self.cdf_tpsa.to_unit_interval(tpsas)
        cdfd_logp = self.cdf_logp.to_unit_interval(logps)
        embedd_tpsa = safe_embed_scalar(
            torch.tensor(cdfd_tpsa), embedding_dim=self.c_embed_dim
        )
        embedd_logp = safe_embed_scalar(
            torch.tensor(cdfd_logp), embedding_dim=self.c_embed_dim
        )
        data = {
            "SMILES": smiles,
            "emb_smiles": embs,
            "tpsa": tpsas,
            "logp": logps,
            "cdfd_tpsa": cdfd_tpsa,
            "cdfd_logp": cdfd_logp,
            "embedded_tpsa": embedd_tpsa,
            "embedded_logp": embedd_logp,
        }
        return data

    def smiles_to_coati_data(self, smiles):
        toks, _ = self.tokenizer.batch_smiles(smiles)
        embs = (
            self.encoder.encode_tokens(toks.to(self.device), self.tokenizer)
            .cpu()
            .numpy()
        )
        tpsas = [
            (
                self.get_tpsa(Chem.MolFromSmiles(smi))
                if Chem.MolFromSmiles(smi) is not None
                else 0.0
            )
            for smi in smiles
        ]
        logps = [
            (
                self.get_logp(Chem.MolFromSmiles(smi))
                if Chem.MolFromSmiles(smi) is not None
                else 0.0
            )
            for smi in smiles
        ]
        cdfd_logps = self.cdf_logp.to_unit_interval(logps)
        cdfd_tpsas = self.cdf_tpsa.to_unit_interval(tpsas)
        embedd_logps = safe_embed_scalar(
            torch.tensor(cdfd_logps), embedding_dim=self.c_embed_dim
        )
        embedd_tpsa = safe_embed_scalar(
            torch.tensor(cdfd_tpsas), embedding_dim=self.c_embed_dim
        )
        data = []
        for smi, emb, logp, cdfd_logp, tpsa, cdfd_tpsa in zip(
            smiles, embs, logps, cdfd_logps, tpsas, cdfd_tpsas
        ):
            data.append(
                {
                    "SMILES": smi,
                    "emb_smiles": emb,
                    "logp": logp,
                    "cdfd_logp": cdfd_logp,
                    "tpsa": tpsa,
                    "cdfd_tpsa": cdfd_tpsa,
                }
            )
        return data


if __name__ == "__main__":
    from coatiLDM.constants import COATI2_DOCS

    preprocessor = ZINC_Preprocessor(
        "./datasets/ZINC22",
        "./datasets/ZINC22_preprocessed_logp_tpsa",
        COATI2_DOCS,
        batch_size=1024,
        c_embed_dim=32,
    )
