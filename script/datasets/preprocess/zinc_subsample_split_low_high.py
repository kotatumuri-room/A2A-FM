import torch
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import itertools
from rdkit.Chem import QED
from rdkit import Chem

from coatiLDM.common.utils import batch_iterable
from coatiLDM.data.transforms import cdf, safe_embed_scalar
from coatiLDM.models.coati.io import load_coati2
from coatiLDM.constants import COATI2_DOCS


def splitparN(iterable, N=3):
    for i, item in itertools.groupby(enumerate(iterable), lambda x: x[0] // N):
        yield (x[1] for x in item)
def get_qeds(smis):
    qeds = np.array([QED.qed(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) is not None else 100 for smi in smis ])
    return qeds[qeds<=1], smis[qeds<=1]
def get_cdfd_qeds(qeds,cdfd):
    cdfd_qed = cdfd.to_unit_interval(qeds)
    return cdfd_qed
def construct_cdf(files_list, num_samples=10, min_size=10, chunk_size=100000):
        sampled_files = np.random.choice(
            np.array(files_list, dtype=str), size=num_samples, replace=True
        )
        qed_samples = []
        for file in tqdm(sampled_files):
            df = np.load(file)
            qed_samples.append(get_qeds(df["SMILES"])[0])
        qed_samples = np.concatenate(qed_samples,axis=0)
        print(
            "qed max:",
            qed_samples.max(),
            "qed min:",
            qed_samples.min(),
        )
        return cdf(qed_samples, min_bin=0, max_bin=1, verbose=True)
def embed_qed(cdfd_qeds,embed_dim):
    return safe_embed_scalar(
            torch.tensor(cdfd_qeds), embedding_dim=embed_dim
        )


def smiles_to_coati_data( smiles,tokenizer,encoder):
        toks, _ = tokenizer.batch_smiles(smiles)
        embs = (
            encoder.encode_tokens(toks.to("cuda"), tokenizer)
            .cpu()
            .numpy()
        )
        return embs

def main(input_dir, save_dir, save_dir_low):
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    save_dir_low = Path(save_dir_low)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir_low.mkdir(exist_ok=True, parents=True)
    files = list(input_dir.rglob("*.npz"))[:488]
    model_doc = COATI2_DOCS["qed_doc"]
    encoder, tokenizer = load_coati2(
        model_doc, freeze=True, device="cuda"
    )
    batch_size = 1024
    qeds, smis, embs, cdfd_qeds, embedded_qeds = [], [], [], [], []
    i = 0
    j = 0
    num_mol = 0
    if (save_dir / "cond_cdfs.pkl").exists():
        cdf = pickle.load(open(save_dir / "cond_cdfs.pkl", "rb"))[0]
    else:
        cdf = construct_cdf(files,num_samples=100)
        with open(save_dir / "cond_cdfs.pkl", "wb") as f:
                    pickle.dump([cdf], f)
        with open(save_dir_low / "cond_cdfs.pkl", "wb") as f:
                    pickle.dump([cdf], f)
    for file in tqdm(files):
        df = np.load(file)
        qed,smi = get_qeds(df["SMILES"])
        smis.append(smi)
        embs.append(smiles_to_coati_data(smi,tokenizer,encoder))
        qeds.append(qed)
        cdfd_qeds.append(get_cdfd_qeds(qed,cdf))
        embedded_qeds.append(embed_qed(cdfd_qeds[-1],32))

        qed_screen = np.concatenate(qeds, axis=0) >= 0.9
        if np.sum(qed_screen) >= batch_size:
            qeds = np.concatenate(qeds, axis=0)
            smis = np.concatenate(smis, axis=0)
            embs = np.concatenate(embs, axis=0)
            cdfd_qeds = np.concatenate(cdfd_qeds, axis=0)
            embedded_qeds = np.concatenate(embedded_qeds, axis=0)
            for dat in batch_iterable(
                zip(smis, embs, qeds, cdfd_qeds, embedded_qeds),
                n=batch_size,
            ):
                data = {
                    "SMILES": np.array([d[0] for d in dat]),
                    "emb_smiles": np.stack([d[1] for d in dat], axis=0),
                    "qed": np.array([d[2] for d in dat]),
                    "cdfd_qed": np.array([d[3] for d in dat]),
                    "embedded_qed": np.stack([d[4] for d in dat], axis=0),
                }
                np.savez(save_dir_low / f"batch_{j}", **data)
                j += 1
            idx = np.random.randint(
                np.sum(qed_screen),
                size=np.sum(np.logical_not(qed_screen)),
            )
            qeds[np.logical_not(qed_screen)] = qeds[qed_screen][idx]
            smis[np.logical_not(qed_screen)] = smis[qed_screen][idx]
            embs[np.logical_not(qed_screen)] = embs[qed_screen][idx]
            cdfd_qeds[np.logical_not(qed_screen)] = cdfd_qeds[qed_screen][idx]
            embedded_qeds[np.logical_not(qed_screen)] = embedded_qeds[qed_screen][idx]
            for dat in batch_iterable(
                zip(smis, embs, qeds, cdfd_qeds, embedded_qeds), n=batch_size
            ):
                num_mol += len(dat)
                data = {
                    "SMILES": np.array([d[0] for d in dat]),
                    "emb_smiles": np.stack([d[1] for d in dat], axis=0),
                    "qed": np.array([d[2] for d in dat]),
                    "cdfd_qed": np.array([d[3] for d in dat]),
                    "embedded_qed": np.stack([d[4] for d in dat], axis=0),
                }
                np.savez(save_dir / f"batch_{i}", **data)
                i += 1
            print(num_mol)
            qeds, smis, embs, cdfd_qeds, embedded_qeds = [], [], [], [], []


if __name__ == "__main__":
    input_dir = (
        "./datasets/ZINC22_preprocessed_logp_tpsa"
    )
    save_dir = "./datasets/ZINC22_preprocessed_highqed_subsampled_b1024"
    save_dir_low = "./datasets/ZINC22_preprocessed_lowqed_subsampled_b1024"
    main(input_dir, save_dir, save_dir_low)

