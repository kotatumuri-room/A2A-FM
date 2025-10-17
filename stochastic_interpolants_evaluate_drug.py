import hydra
from hydra.utils import instantiate
from stochastic_interpolant.vector_field import VelocityField
from stochastic_interpolant.path_optimizer import PathOptimizer
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from itertools import combinations
from stochastic_interpolant.utils import Velocity
from torchdiffeq import odeint
import torch
import numpy as np
import time
from tqdm.auto import tqdm,trange
import pickle

from rdkit.Chem import Draw, AllChem, QED, MolFromSmiles, Descriptors
from rdkit import Chem
from rdkit import DataStructs
from coatiLDM.data.decoding import force_decode_valid_batch_efficient
from coatiLDM.data.transforms import safe_embed_scalar
from coatiLDM.models.coati.io import load_coati2


def get_filename(logdir):
    versions = sorted(list(logdir.glob("version_*")))
    if len(versions) == 0:
        versions.append(logdir / "version_0")
    latest_number = int(versions[-1].name[8:])
    return logdir / f"version_{latest_number+1}"

def prepare_vels(gs,root_dir,config):
    K = config.eval.K
    alphas = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i==j:
                continue
            alphas[i][j]=get_vel(gs,i,j,root_dir,config)
    return alphas


def get_vel(gs,i,j,root_dir,config):
    reverse = False
    if j < i:
        reverse = True
        tmp = i
        i = j
        j = tmp
    ckpt_dir = root_dir / f"alpha_{i}_{j}"
    print(ckpt_dir,list(ckpt_dir.rglob("*.ckpt")))
    ckpt_path = sorted(list(ckpt_dir.rglob("*.ckpt")))[0]
    print(f"using check point {ckpt_path}")
    alpha = PathOptimizer.load_from_checkpoint(ckpt_path, myconfig=config.alpha)
    vel = Velocity(gs, alpha,reverse = reverse).cpu()
    return vel
def prepare_gs(ckpt_dirs_vf,config):
    gs = []
    gt_versions = [-1] * len(ckpt_dirs_vf)
    if hasattr(config, "ckpt_versions_gt"):
        gt_versions = config.ckpt_versions_gt
        print(gt_versions)

    for ckpt_dir, v in zip(ckpt_dirs_vf, gt_versions):
        print((ckpt_dir/"lightning_logs"/f"version_{v}"/"checkpoints"))
        ckpt_path = list((ckpt_dir/"lightning_logs"/f"version_{v}"/"checkpoints").glob("*.ckpt"))[0]
        print(f"using check point {ckpt_path}")
        gs.append(
            VelocityField.load_from_checkpoint(ckpt_path, myconfig=config.vector_field)
        )
    return gs
@torch.no_grad()
def integrate(vels,initx,initc,targc,ts,kmeans_model,l0,targlabel,config):
    b, d = initx.shape
    traj = initx.expand(len(ts),b,d)
    i = int(kmeans_model.predict(l0.reshape(1,2)).squeeze())
    js = torch.from_numpy(kmeans_model.predict(targlabel.cpu().numpy().astype(float))).to(config.eval.device)
    for j in range(config.eval.K):
        x0 = initx[js==j]
        if len(x0)==0:
            continue
        if i == j:
            continue
        vel = vels[i][j].cuda()
        traj[:,js==j,:]=odeint(vel,x0,ts).squeeze(1)
    return traj

@hydra.main(
    config_path="stochastic_interpolants_config",
    config_name="stochastic_interpolants",
    version_base=None,
)
def validate(config):
    val_datasets = instantiate(config.val_datasets, _recursive_=False)
    K = len(val_datasets)
    root_dir = Path(config.savedir)
    ckpt_dirs_vf = [root_dir / f"g{k}" for k in range(K)]
    c_list = config.c_list
    assert len(c_list) == K
    gs = prepare_gs(ckpt_dirs_vf,config)
    kmeans_model = val_datasets.kmeans_model

    validation_conds = torch.tensor(config.eval.validation_conds)
    cond_cdf = pickle.load(open(config.eval.cond_cdf_path, "rb"))
    cdf_names = ["cdf_logp", "cdf_tpsa"]
    c_emb_name = ["logp", "tpsa"]
    validation_conds_emb=embed_condition(
            validation_conds.reshape(-1, 2),cond_cdf,cdf_names,cdim=32
    )
    initx, initc, initmols,initlabel = get_valset(config.eval.initial_point_dir,"emb_smiles",c_emb_name,cond_cdf,cdf_names)
    initx, initc = torch.from_numpy(initx).to(config.eval.device), torch.from_numpy(
        initc
    ).to(config.eval.device)
    targc = validation_conds_emb
    targc = targc.repeat(config.eval.num_seed, 1).to(config.eval.device)
    targlabels = validation_conds.repeat(config.eval.num_seed, 1)
    coati, tokenizer = load_coati(config.eval.autoencoder_path, config.eval.device)
    closs = np.zeros(len(initx))
    csim = np.zeros(len(initx))
    savedir = Path(config.savedir) / (
        "full_attempts" + f"{time.time()/1000000:.5f}"[5:] + ".npz"
    )
    savedir.touch(exist_ok=False)
    full_attempts = {}
    pb = tqdm(enumerate(zip(initx, initc, initmols,initlabel)), total=len(initx))
    ts = torch.linspace(0,1,config.eval.timesteps,device=config.eval.device)
    vels = prepare_vels(gs,root_dir,config)
    for k, (x0, c0, mols0,l0) in pb:
        x0_ = x0.unsqueeze(0).expand(len(targc), -1)
        c0_ = c0.unsqueeze(0).expand_as(targc)
        x0_ = x0_ + torch.randn_like(x0_) * config.eval.noise_intensity
        traj = integrate(vels,x0_,c0_,targc,ts,kmeans_model,l0,targlabels,config)
        mols = decode(traj[-1], coati, tokenizer, config.eval.device)
        diff_tpsa_logp = np.zeros(len(validation_conds))
        tanimoto_sims = np.zeros(len(validation_conds))
        attempt = {}
        for i, c1 in enumerate(validation_conds):
            trials = np.array(mols[i :: len(validation_conds_emb)])
            tpsa = get_tpsa(trials)
            logp = get_logp(trials)
            success = (tpsa != None) & (logp != None)
            if np.sum(success) == 0:
                print(
                    f"No success! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                )
                continue
            conds = cdf_condition(np.stack([logp, tpsa], axis=-1)[success],cond_cdf,cdf_names)
            tanimoto_sim = get_tanimoto_sim(
                trials[success], np.array(mols0).repeat(len(success))
            )
            if np.sum(tanimoto_sim<1) == 0:
                print(
                    f"No changes! initx:{Chem.MolToSmiles(mols0)} c1:{c1.numpy()}"
                )
                continue
            loss = np.linalg.norm(
                conds[tanimoto_sim < 1]
                - cdf_condition(c1.cpu().reshape(1, -1).numpy(),cond_cdf,cdf_names),
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


    
def get_tpsa(mols):
    return np.array(
        [Descriptors.TPSA(mol) if mol is not None else None for mol in mols]
    )

def get_logp(mols):
    return np.array(
        [Chem.Crippen.MolLogP(mol) if mol is not None else None for mol in mols]
    )

def get_tanimoto_sim(mol1, mol2):
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

def get_valset(initial_point_dir: str,emb_name,c_emb_name,cond_cdf,cdf_names):
    with open(initial_point_dir, "rb") as f:
        val_set = np.load(f)
        x = val_set[emb_name]
        c = []
        for c_name in c_emb_name:
            c.append(val_set[c_name])
        c = np.stack(c, axis=1)
        c_emb = embed_condition(torch.from_numpy(c),cond_cdf,cdf_names,cdim=32)
        smis = val_set["SMILES"]
        mols = [MolFromSmiles(smi) for smi in smis]
    return x, c_emb.cpu().numpy(), mols,c

def cdf_condition(conds: np.ndarray,cond_cdf,cdf_names):
    assert isinstance(conds, np.ndarray)
    assert conds.ndim == 2
    embed_out = []
    for i in range(conds.shape[1]):
        cdf_c = cond_cdf[cdf_names[i]].cdf(conds[:, i])
        embed_out.append(cdf_c)
    return np.stack(embed_out, axis=-1)

def load_coati(autoencoder_path, device):
    coati, tokenizer = load_coati2(
        autoencoder_path,
        device,
        freeze=True,
        old_architecture=False,
        force_cpu=False,  # needed to deserialize on some cpu-only machines
    )
    return coati, tokenizer

def embed_condition(conds: torch.Tensor,cond_cdf,cdf_names,cdim):
    assert conds.dim() == 2
    assert conds.shape[1] == 2
    embed_out = []
    for i in range(2):
        cdf_c = torch.from_numpy(
            cond_cdf[cdf_names[i]].cdf(conds[:, i].cpu().numpy())
        )
        embed_c = safe_embed_scalar(cdf_c,cdim // 2).to(
            conds.device
        )
        embed_out.append(embed_c)
    return torch.cat(embed_out, dim=-1)

def decode(
    sample_batch: torch.Tensor, coati, tokenizer, device, batch_size=100
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

        



if __name__ == "__main__":
    validate()
