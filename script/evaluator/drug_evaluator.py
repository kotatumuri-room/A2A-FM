from script.evaluator.base_evaluator import BaseEvaluator
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm, trange
from rdkit.Chem import Draw, AllChem, QED, MolFromSmiles, Descriptors
from rdkit import Chem
from rdkit import DataStructs
from coatiLDM.data.decoding import force_decode_valid_batch_efficient
from coatiLDM.data.transforms import safe_embed_scalar
from coatiLDM.models.coati.io import load_coati2
from typing import List
from pathlib import Path

from torchdiffeq import odeint
import time
from torch.nn.functional import mse_loss


def get_qed(mol):
    try:
        return QED.qed(mol)
    except Chem.rdchem.KekulizeException:
        print("qed fail!")
        return 0.0


def measure_tanimoto_similarity_success_rate(mol1, mol2):
    assert len(mol1) == len(mol2), f"mol1 {len(mol1)}, mol2 {len(mol2)}"
    qeds = np.array([get_qed(mol) for mol in mol2])
    mask_qed = qeds >= 0.9
    morgan_fp1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mol1]
    morgan_fp2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mol2]
    tanimoto = np.array(
        [
            DataStructs.TanimotoSimilarity(morgan_fp1[i], morgan_fp2[i])
            for i in range(len(morgan_fp1))
        ]
    )
    mask_tanimoto = tanimoto >= 0.4
    mask_tanimoto1 = tanimoto == 1.0
    total_mask = mask_qed & mask_tanimoto
    return tanimoto, qeds, total_mask, mask_tanimoto1


class Zinc22Evaluator(BaseEvaluator):

    def __init__(
        self,
        autoencoder_path: str,
        cond_cdf_path: str,
        initial_point_dir: str,
        target_trial_conds: List,
        validation_conds: List,
        device: str = "cuda",
        emb_name: str = "emb_smiles",
        c_emb_name: str = "qed",
        noise_intensity_list: List[float] = [
            0.0,
        ],
        boosting_list: List[float] = [
            1.0,
        ],
        num_seed: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_name = emb_name
        self.c_emb_name = c_emb_name
        self.autoencoder_path = autoencoder_path
        self.device = device  # for evaluation not validation
        self.cond_cdf = pickle.load(open(cond_cdf_path, "rb"))
        self.initial_point_dir = initial_point_dir
        self.target_trial_conds = target_trial_conds
        self.validation_conds_emb = self.embed_condition(
            torch.tensor(validation_conds).reshape(-1, 1)
        )
        self.validation_conds = torch.tensor(validation_conds)
        self.boosting_list = boosting_list
        self.noise_intensity_list = noise_intensity_list
        self.num_seed = num_seed

    def validation(self, batch: torch.Tensor) -> dict:
        initx, initc, _, _ = batch
        output_dict = {}
        device = initx.device
        coati, tokenizer = self.load_coati(self.autoencoder_path, device)
        for targc, c in zip(self.validation_conds_emb, self.validation_conds):
            traj = self.integrate(
                initx, targc.unsqueeze(0).expand_as(initc).to(device), initc
            )
            mols = self.decode(traj[-1], coati, tokenizer, device)
            qeds = torch.tensor([QED.qed(mol) for mol in mols if mol is not None])
            fail_ratio = np.array(
                [1 if mol is None else 0 for mol in mols]
            ).sum() / len(initx)
            output_dict[f"c loss {c:.02f}"] = mse_loss(qeds, c.expand_as(qeds))
            output_dict[f"fail ratio {c:.02f}"] = fail_ratio
        return output_dict

    def get_tanimoto_valset(self, initial_point_dir: str):
        with open(initial_point_dir, "rb") as f:
            val_set = pickle.load(f)
            x = np.stack([v[self.emb_name] for v in val_set], axis=0)
            c = np.stack([v[self.c_emb_name] for v in val_set], axis=0).reshape(-1, 1)
            c_emb = self.embed_condition(torch.from_numpy(c))
            mols = np.array([MolFromSmiles(v["SMILES"]) for v in val_set])
        return x, c_emb.cpu().numpy(), mols

    def load_coati(self, autoencoder_path, device):
        coati, tokenizer = load_coati2(
            autoencoder_path,
            device,
            freeze=True,
            old_architecture=False,
            force_cpu=False,  # needed to deserialize on some cpu-only machines
        )
        return coati, tokenizer

    def embed_condition(self, conds: torch.Tensor):
        assert isinstance(conds, torch.Tensor)
        assert conds.dim() == 2
        embed_out = []
        cdf_c = torch.from_numpy(self.cond_cdf[0].cdf(conds[:, 0].cpu().numpy()))
        embed_c = safe_embed_scalar(cdf_c, self.data_handler.c_dim).to(conds.device)
        embed_out.append(embed_c)
        return torch.cat(embed_out, dim=-1)

    def decode(
        self, sample_batch: torch.Tensor, coati, tokenizer, device, batch_size=100
    ):
        n = len(sample_batch)
        smis = []
        if n > batch_size:
            for i in trange(n // batch_size, leave=False, desc="decoding"):
                smis_batch = force_decode_valid_batch_efficient(
                    sample_batch[i * batch_size : (i + 1) * batch_size].to(device),
                    coati,
                    tokenizer,
                    max_attempts=5,
                    inv_temp=1.5,
                    k=500,
                    noise_scale=0.0,
                    chiral=True,
                    silent=True,
                )
                smis = smis + smis_batch
            smis_batch = force_decode_valid_batch_efficient(
                sample_batch[(i + 1) * batch_size :].to(device),
                coati,
                tokenizer,
                max_attempts=5,
                inv_temp=1.5,
                k=500,
                noise_scale=0.0,
                chiral=True,
                silent=True,
            )
            smis = smis + smis_batch
        else:
            smis = force_decode_valid_batch_efficient(
                sample_batch.to(device),
                coati,
                tokenizer,
                max_attempts=5,
                inv_temp=1.5,
                k=500,
                noise_scale=0.0,
                chiral=True,
                silent=True,
            )
        molecules = [
            MolFromSmiles(smiles) if smiles != "" else None
            for smiles in tqdm(smis, desc="mol to smiles", leave=False)
        ]
        return molecules

    def evaluate(self):
        self.run_drug_exploration_effective(
            initial_point_dir=self.initial_point_dir,
            trial_targc=torch.tensor(self.target_trial_conds),
            num_seed=self.num_seed,
        )

    def run_drug_exploration_effective(
        self, trial_targc: torch.Tensor, initial_point_dir: str, num_seed: int = 1
    ):
        MAX_ORACLE_CALLS = 50000
        x0, c0, mols0 = self.get_tanimoto_valset(initial_point_dir)
        full_attempts = {"starting_points": {"mol": mols0, "qed": c0}}
        x0, c0 = torch.from_numpy(x0), torch.from_numpy(c0)
        coati, tokenizer = self.load_coati(self.autoencoder_path, device=self.device)
        num_found = 0
        path_attempts = Path(self.config.savedir) / (
            "full_attempts" + f"{time.time()/1000000:.5f}"[5:] + ".pkl"
        )
        path_attempts.touch(exist_ok=False)
        print(f"saving at {path_attempts}")
        max_oracle_call = 0
        for i, (initx, initc, initmol) in tqdm(
            enumerate(zip(x0, c0, mols0)), total=len(x0), desc="evaluating qed"
        ):
            oracle_calls = 0
            found = False
            while oracle_calls < MAX_ORACLE_CALLS and not found:
                for boosting in self.boosting_list:
                    for targc in trial_targc:
                        noise = torch.cat(
                            [
                                e
                                * torch.randn(num_seed, len(initx), device=self.device)
                                for e in self.noise_intensity_list
                            ],
                            dim=0,
                        )
                        noise_intensities = np.concatenate(
                            [e * np.ones(num_seed) for e in self.noise_intensity_list],
                            axis=0,
                        )
                        x0_ = (
                            initx.unsqueeze(0).expand_as(noise).to(self.device) + noise
                        )
                        c0_ = (
                            initc.unsqueeze(0)
                            .expand(len(x0_), len(initc))
                            .to(self.device)
                        )
                        c1_ = targc.unsqueeze(0).expand_as(c0_).to(self.device)
                        traj = self.integrate(x0_, c1_, c0_, boosting=boosting)
                        mols = np.array(
                            self.decode(
                                traj[-1], coati, tokenizer, self.device, batch_size=400
                            )
                        )
                        not_none_mask = mols != None
                        mols0_ = (
                            np.array([initmol]).reshape(1, -1).repeat(len(traj[-1]))
                        )
                        tanimotos, qeds, success_mask_fail, tanimoto1 = (
                            measure_tanimoto_similarity_success_rate(
                                mols0_[not_none_mask], mols[not_none_mask]
                            )
                        )
                        attempt = [
                            {
                                "mol": mol,
                                "qed": qed,
                                "tanimoto": tanimoto,
                                "noise_intensity": ni,
                                "success": success,
                            }
                            for mol, qed, tanimoto, ni, success in zip(
                                mols[not_none_mask],
                                qeds,
                                tanimotos,
                                noise_intensities,
                                success_mask_fail,
                            )
                        ]
                        full_attempts[
                            f"initsmiles:{Chem.MolToSmiles(initmol)}-targc:{targc:.3f}-boosting:{boosting:.3f}"
                        ] = attempt
                        oracle_calls += len(c0_)
                        if np.sum(success_mask_fail) > 0:
                            num_found += 1
                            print(f"FOUND {num_found} out of 800")
                            print(f"MISSED {i-num_found + 1} out of 800")
                            print(f"remaining {800-i-1}")
                            print(f"oracle calls {oracle_calls}")
                            max_oracle_call = max(max_oracle_call, oracle_calls)
                            print(f"max oracle calls {max_oracle_call}")
                            found = True
                            break
                        else:
                            print(
                                f"oracle calls {oracle_calls} for {Chem.MolToSmiles(initmol)}"
                            )
                        if oracle_calls >= MAX_ORACLE_CALLS:
                            break
                    if oracle_calls >= MAX_ORACLE_CALLS or found:
                        break
                if found or oracle_calls >= MAX_ORACLE_CALLS:
                    break
                with open(path_attempts, "wb") as f:
                    pickle.dump(full_attempts, f)
        with open(path_attempts, "wb") as f:
            pickle.dump(full_attempts, f)
        print(f"saved at {path_attempts}")

    def run_drug_exploration(
        self, trial_targc: torch.Tensor, initial_point_dir: str, num_seed: int = 1
    ):
        x0, c0, mols0 = self.get_tanimoto_valset(initial_point_dir)
        success_mask = np.zeros(len(c0))
        pb = tqdm(self.boosting_list, desc="evaluating transfer")
        full_attempts = {"starting_points": {"mol": mols0, "qed": c0}}
        coati, tokenizer = self.load_coati(self.autoencoder_path, device=self.device)
        num_attempts = 0
        for boosting in pb:
            for noise_intensity in self.noise_intensity_list:
                for c1 in trial_targc:
                    seeds = (
                        torch.randn(
                            x0.shape[0] * num_seed, x0.shape[1], device=self.device
                        )
                        * noise_intensity
                    )
                    for i in range(num_seed):
                        num_attempts = num_attempts + 1
                        fail_mask = np.logical_not(success_mask)
                        trans_clist0 = torch.from_numpy(c0[fail_mask]).to(self.device)
                        trans_clist1 = (
                            self.embed_condition(c1.reshape(1, -1))
                            .expand_as(trans_clist0)
                            .to(self.device)
                        )
                        initx = torch.from_numpy(x0[fail_mask]).to(self.device)
                        noise = seeds[i * x0.shape[0] : (i + 1) * x0.shape[0]][
                            fail_mask
                        ]
                        initx = initx + noise
                        traj = self.integrate(
                            initx, trans_clist1, trans_clist0, boosting
                        )
                        mols = np.array(
                            self.decode(
                                traj[-1], coati, tokenizer, self.device, batch_size=400
                            )
                        )
                        not_none_mask = mols != None
                        valid = fail_mask
                        valid[fail_mask] = not_none_mask
                        tanimotos, qeds, success_mask_fail, tanimoto1 = (
                            measure_tanimoto_similarity_success_rate(
                                mols0[valid], mols[not_none_mask]
                            )
                        )
                        attempt = np.empty(shape=(len(c0)), dtype=object)
                        attempt[valid] = [
                            {"mol": mol, "qed": qed, "tanimoto": tanimoto}
                            for mol, qed, tanimoto in zip(
                                mols[not_none_mask], qeds, tanimotos
                            )
                        ]
                        full_attempts[
                            f"targc:{c1:.3f}-noise:{noise_intensity:.3f}-boosting:{boosting:.3f}-noise-index:{i}"
                        ] = attempt
                        success_mask[valid] = success_mask_fail
                        pb.set_description(
                            f"qed {c1.cpu().item()} noise {noise_intensity} boosting {boosting} noise index {i} num attempts {num_attempts} both ok (up to now) {np.sum(success_mask)} both ok (current){np.sum(success_mask_fail)} tanimoto ok {np.sum(tanimotos>0.4)} qed ok {np.sum(qeds>0.9)} remaining {800-np.sum(success_mask)} tanimoto=1 {np.sum(tanimoto1)}"
                        )
        path_attempts = Path(self.config.savedir) / (
            "full_attempts" + f"{time.time()/1000000:.5f}"[5:] + ".pkl"
        )
        path_attempts.parent.mkdir(exist_ok=True, parents=True)
        with open(path_attempts, "wb") as f:
            pickle.dump(full_attempts, f)
        print(f"saved attempts to {path_attempts}")


class ZINC22_2DimEvaluator(BaseEvaluator):

    def __init__(
        self,
        autoencoder_path: str,
        cond_cdf_path: str,
        initial_point_dir: str,
        validation_conds: List,  # LOGP FIRST
        device: str = "cuda",
        emb_name: str = "emb_smiles",
        c_emb_name: str = ["logp", "tpsa"],
        num_seed: int = 1,
        noise_intensity: float = 0.0,
        cdf_names: List = ["cdf_logp", "cdf_tpsa"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb_name = emb_name
        self.c_emb_name = c_emb_name
        self.autoencoder_path = autoencoder_path
        self.device = device  # for evaluation not validation
        self.cond_cdf = pickle.load(open(cond_cdf_path, "rb"))
        self.cdf_names = ["cdf_logp", "cdf_tpsa"]
        self.initial_point_dir = initial_point_dir
        self.validation_conds_emb = self.embed_condition(
            torch.tensor(validation_conds).reshape(-1, 2)
        )
        self.validation_conds = torch.tensor(validation_conds)
        self.num_seed = num_seed
        self.noise_intensity = noise_intensity

    def validation(self, batch, batch_idx):
        pass

    def evaluate(self):
        initx, initc, initmols = self.get_valset(self.initial_point_dir)
        initx, initc = torch.from_numpy(initx).to(self.device), torch.from_numpy(
            initc
        ).to(self.device)
        targc = self.validation_conds_emb
        targc = targc.repeat(self.num_seed, 1).to(self.device)
        coati, tokenizer = self.load_coati(self.autoencoder_path, self.device)
        closs = np.zeros(len(initx))
        csim = np.zeros(len(initx))
        savedir = Path(self.config.savedir) / (
            "full_attempts" + f"{time.time()/1000000:.5f}"[5:] + ".npz"
        )
        savedir.touch(exist_ok=False)
        full_attempts = {}
        pb = tqdm(enumerate(zip(initx, initc, initmols)), total=len(initx))
        for k, (x0, c0, mols0) in pb:
            x0_ = x0.unsqueeze(0).expand(len(targc), -1)
            c0_ = c0.unsqueeze(0).expand_as(targc)
            x0_ = x0_ + torch.randn_like(x0_) * self.noise_intensity
            traj = self.integrate(x0_, targc, c0_)
            mols = self.decode(traj[-1], coati, tokenizer, self.device)
            diff_tpsa_logp = np.zeros(len(self.validation_conds))
            tanimoto_sims = np.zeros(len(self.validation_conds))
            attempt = {}
            for i, c1 in enumerate(self.validation_conds):
                trials = np.array(mols[i :: len(self.validation_conds_emb)])
                tpsa = self.get_tpsa(trials)
                logp = self.get_logp(trials)
                success = (tpsa != None) & (logp != None)
                if np.sum(success) == 0:
                    print(
                        f"No success! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                    )
                    continue
                conds = self.cdf_condition(np.stack([logp, tpsa], axis=-1)[success])
                tanimoto_sim = self.get_tanimoto_sim(
                    trials[success], np.array(mols0).repeat(len(success))
                )
                if np.sum(tanimoto_sim<1) == 0:
                    print(
                        f"No changes! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                    )
                    continue
                loss = np.linalg.norm(
                    conds[tanimoto_sim < 1]
                    - self.cdf_condition(c1.cpu().reshape(1, -1).numpy()),
                    axis=1,
                )
                diff_tpsa_logp[i] = np.min(loss)  # LOGP FIRST
                tanimoto_sims[i] = tanimoto_sim[tanimoto_sim < 1][np.argmin(loss)]
                attempt[f"target:{np.round(c1.cpu().numpy(),3)}"] = {
                    "mols": trials,
                    "success": success,
                    "tpsa": tpsa,
                    "logp": logp,
                    "tanimoto": tanimoto_sim,
                }

            closs[k] = diff_tpsa_logp.mean()
            csim[k] = tanimoto_sims.mean()
            pb.set_description(
                f"init mol:{Chem.MolToSmiles(mols0)}  diff_tpsa_logp: {diff_tpsa_logp.mean():.03f}/{closs[:k+1].mean():.03f} tanimoto {tanimoto_sim.mean():.03f}/{csim[:k+1].mean():.03f}"
            )
            full_attempts[f"init mol:{Chem.MolToSmiles(mols0)}"] = {
                "attempts": attempt,
                "closs": closs[k],
                "similarity mean": csim[k],
            }
            with open(savedir, "wb") as f:
                pickle.dump(full_attempts, f)

    def get_tpsa(self, mols):
        return np.array(
            [Descriptors.TPSA(mol) if mol is not None else None for mol in mols]
        )

    def get_logp(self, mols):
        return np.array(
            [Chem.Crippen.MolLogP(mol) if mol is not None else None for mol in mols]
        )

    def get_tanimoto_sim(self, mol1, mol2):
        morgan_fp1 = [
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mol1
        ]
        morgan_fp2 = [
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mol2
        ]
        tanimoto = np.array(
            [
                DataStructs.TanimotoSimilarity(morgan_fp1[i], morgan_fp2[i])
                for i in range(len(morgan_fp1))
            ]
        )
        return tanimoto

    def get_valset(self, initial_point_dir: str):
        with open(initial_point_dir, "rb") as f:
            val_set = np.load(f)
            x = val_set[self.emb_name]
            c = []
            for c_name in self.c_emb_name:
                c.append(val_set[c_name])
            c = np.stack(c, axis=1)
            c_emb = self.embed_condition(torch.from_numpy(c))
            smis = val_set["SMILES"]
            mols = [MolFromSmiles(smi) for smi in smis]
        return x, c_emb.cpu().numpy(), mols

    def embed_condition(self, conds: torch.Tensor):
        assert isinstance(conds, torch.Tensor)
        assert conds.dim() == 2
        embed_out = []
        for i in range(conds.shape[1]):
            cdf_c = torch.from_numpy(
                self.cond_cdf[self.cdf_names[i]].cdf(conds[:, i].cpu().numpy())
            )
            embed_c = safe_embed_scalar(cdf_c, self.data_handler.c_dim).to(conds.device)
            embed_out.append(embed_c)
        return torch.cat(embed_out, dim=-1)

    def cdf_condition(self, conds: np.ndarray):
        assert isinstance(conds, np.ndarray)
        assert conds.ndim == 2
        embed_out = []
        for i in range(conds.shape[1]):
            cdf_c = self.cond_cdf[self.cdf_names[i]].cdf(conds[:, i])
            embed_out.append(cdf_c)
        return np.stack(embed_out, axis=-1)

    def load_coati(self, autoencoder_path, device):
        coati, tokenizer = load_coati2(
            autoencoder_path,
            device,
            freeze=True,
            old_architecture=False,
            force_cpu=False,  # needed to deserialize on some cpu-only machines
        )
        return coati, tokenizer

    def embed_condition(self, conds: torch.Tensor):
        assert isinstance(conds, torch.Tensor)
        assert conds.dim() == 2
        assert conds.shape[1] == 2
        embed_out = []
        for i in range(2):
            cdf_c = torch.from_numpy(
                self.cond_cdf[self.cdf_names[i]].cdf(conds[:, i].cpu().numpy())
            )
            embed_c = safe_embed_scalar(cdf_c, self.data_handler.c_dim // 2).to(
                conds.device
            )
            embed_out.append(embed_c)
        return torch.cat(embed_out, dim=-1)

    def decode(
        self, sample_batch: torch.Tensor, coati, tokenizer, device, batch_size=100
    ):
        n = len(sample_batch)
        smis = []
        if n > batch_size:
            for i in trange(n // batch_size, leave=False, desc="decoding"):
                smis_batch = force_decode_valid_batch_efficient(
                    sample_batch[i * batch_size : (i + 1) * batch_size].to(device),
                    coati,
                    tokenizer,
                    max_attempts=5,
                    inv_temp=1.5,
                    k=500,
                    noise_scale=0.0,
                    chiral=True,
                    silent=True,
                )
                smis = smis + smis_batch
            smis_batch = force_decode_valid_batch_efficient(
                sample_batch[(i + 1) * batch_size :].to(device),
                coati,
                tokenizer,
                max_attempts=5,
                inv_temp=1.5,
                k=500,
                noise_scale=0.0,
                chiral=True,
                silent=True,
            )
            smis = smis + smis_batch
        else:
            smis = force_decode_valid_batch_efficient(
                sample_batch.to(device),
                coati,
                tokenizer,
                max_attempts=5,
                inv_temp=1.5,
                k=500,
                noise_scale=0.0,
                chiral=True,
                silent=True,
            )
        molecules = [
            MolFromSmiles(smiles) if smiles != "" else None
            for smiles in tqdm(smis, desc="mol to smiles", leave=False)
        ]
        return molecules


class ZINC22_2DimCFGDiffusionEvaluator(ZINC22_2DimEvaluator):
    def __init__(
        self, timesteps_eval: float = 0.5, guidance_weight: float = 0.3, **kwargs
    ):
        super().__init__(**kwargs)
        self.timesteps_eval = timesteps_eval
        self.guidance_weight = guidance_weight

    def evaluate(self):
        initx, initc, initmols = self.get_valset(self.initial_point_dir)
        initx, initc = torch.from_numpy(initx).to(self.device), torch.from_numpy(
            initc
        ).to(self.device)
        targc = self.validation_conds_emb
        targc = targc.repeat(self.num_seed, 1).to(self.device)
        coati, tokenizer = self.load_coati(self.autoencoder_path, self.device)
        closs = np.zeros(len(initx))
        csim = np.zeros(len(initx))
        savedir = Path(self.config.savedir) / (
            "full_attempts" + f"{time.time()/1000000:.5f}"[5:] + ".npz"
        )
        savedir.touch(exist_ok=False)
        full_attempts = {}
        pb = tqdm(enumerate(zip(initx, initc, initmols)), total=len(initx))
        for k, (x0, c0, mols0) in pb:
            x0_ = x0.unsqueeze(0).expand(len(targc), -1)
            traj = self.partial_diffusion(
                x0_, targc, timesteps=self.timesteps_eval, w=self.guidance_weight
            )
            mols = self.decode(traj[-1], coati, tokenizer, self.device)
            diff_tpsa_logp = np.zeros(len(self.validation_conds))
            tanimoto_sims = np.zeros(len(self.validation_conds))
            attempt = {}
            for i, c1 in enumerate(self.validation_conds):
                trials = np.array(mols[i :: len(self.validation_conds_emb)])
                tpsa = self.get_tpsa(trials)
                logp = self.get_logp(trials)
                success = (tpsa != None) & (logp != None)
                if np.sum(success) == 0:
                    print(
                        f"No success! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                    )
                    continue
                conds = self.cdf_condition(np.stack([logp, tpsa], axis=-1)[success])
                tanimoto_sim = self.get_tanimoto_sim(
                    trials[success], np.array(mols0).repeat(len(success))
                )
                if np.sum(tanimoto_sim < 1) == 0:
                    print(
                        f"No changes! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                    )
                    continue
                loss = np.linalg.norm(
                    conds[tanimoto_sim < 1]
                    - self.cdf_condition(c1.cpu().reshape(1, -1).numpy()),
                    axis=1,
                )
                diff_tpsa_logp[i] = np.min(loss)  # LOGP FIRST
                tanimoto_sims[i] = tanimoto_sim[tanimoto_sim < 1][np.argmin(loss)]
                attempt[f"target:{np.round(c1.cpu().numpy(),3)}"] = {
                    "mols": trials,
                    "success": success,
                    "tpsa": tpsa,
                    "logp": logp,
                    "tanimoto": tanimoto_sim,
                }

            closs[k] = diff_tpsa_logp.mean()
            csim[k] = tanimoto_sims.mean()
            pb.set_description(
                f"init mol:{Chem.MolToSmiles(mols0)}  diff_tpsa_logp: {diff_tpsa_logp.mean():.03f}/{closs[:k+1].mean():.03f} tanimoto {tanimoto_sim.mean():.03f}/{csim[:k+1].mean():.03f}"
            )
            full_attempts[f"init mol:{Chem.MolToSmiles(mols0)}"] = {
                "attempts": attempt,
                "closs": closs[k],
                "similarity mean": csim[k],
            }
            with open(savedir, "wb") as f:
                pickle.dump(full_attempts, f)
