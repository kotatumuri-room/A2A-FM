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
from matplotlib import pyplot as plt
from lightning.pytorch import loggers as pl_loggers
from torchvision.utils import make_grid, save_image
import ot
from omegaconf import DictConfig
import yaml
from script.datasets.preprocess.embed_celeba_hq import load_model_from_config
from script.datasets.CelebAHQ_dataset import CLASS_NAMES
import math
import cv2



def get_filename(logdir):
    versions = sorted(list(logdir.glob("version_*")))
    if len(versions) == 0:
        versions.append(logdir / "version_0")
    latest_number = int(versions[-1].name[8:])
    return logdir / f"version_{latest_number+1}"

def prepare_vels(gs,root_dir,config,datasets):
    K = config.K
    alphas = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if i==j:
                continue
            alphas[i][j]=get_vel(gs,i,j,root_dir,config,datasets)
    return alphas


def get_vel(gs,i,j,root_dir,config,datasets):
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
    alpha = PathOptimizer.load_from_checkpoint(ckpt_path, myconfig=config.alpha,gs=gs,i=i,j=j,datasets=datasets)
    vel = Velocity(gs, alpha,reverse = reverse)
    return vel
def prepare_gs(ckpt_dirs_vf,config):
    gs = []
    gt_versions = [-1] * len(ckpt_dirs_vf)
    if hasattr(config, "ckpt_versions_gt"):
        gt_versions = config.ckpt_versions_gt
        print(gt_versions)

    for ckpt_dir, v in zip(ckpt_dirs_vf, gt_versions):
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
    i = int(kmeans_model.predict(l0.cpu().numpy().reshape(1,-1)))
    js = torch.from_numpy(kmeans_model.predict(targlabel.cpu().numpy())).to(config.eval.device)
    for j in range(config.K):
        x0 = initx[js==j]
        if len(x0)==0:
            continue
        if i == j:
            continue
        print(i,j)
        vel = vels[i][j]
        traj[:,js==j,:]=odeint(vel,x0,ts,atol=1e-3,rtol=1e-3)
    return traj
def prepare_ae(ae_config,ae_ckpt):
        with open(
            ae_config,
            "rb",
        ) as f:
            config = yaml.safe_load(f)
            config = DictConfig(config)
        model, _ = load_model_from_config(
            config,
            ae_ckpt,
        )
        return model
def load_image(path,c_indecies):
        data = np.load(path)
        x_emb = (
            torch.from_numpy(data["x_emb"])
            .reshape(1, 3 * 64 * 64)
            .cuda()
        )
        x = torch.from_numpy(data["x"])
        c = torch.from_numpy(data["c"][c_indecies]).reshape(1, len(c_indecies)).cuda()
        return x, c, x_emb
@torch.no_grad()
def decode_imgs(ae, latent: torch.Tensor) -> torch.Tensor:
    b, d = latent.shape
    decoder = ae["model"].to(latent.device)
    x = latent.reshape(b, 3, 64, 64)
    batch_size = 16
    img = torch.zeros(b, 3, 256, 256)
    for i in range(0, len(x), batch_size):
        img[i : (i + batch_size)] = decoder.decode(x[i : (i + batch_size)])
    img[i:] = decoder.decode(x[i:])
    img = img.reshape(b,3, 256, 256)
    return img

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

    initial_files = list(Path(config.eval.initial_path).glob("*.npz"))
    savedir = Path(config.eval.img_path)
    savedir.mkdir(exist_ok=True, parents=True)
    ae = prepare_ae(config.eval.ae_config,config.eval.ae_ckpt)
    c_indecies = config.eval.c_indecies
    ts = torch.linspace(0,1,config.eval.timesteps,device="cuda")
    vels = prepare_vels(gs,root_dir,config,val_datasets)
    print(CLASS_NAMES)
    for i, img_path in enumerate(tqdm(initial_files)):
        x, initc, initx = load_image(img_path,c_indecies)
        targc = torch.tensor(config.eval.targc).reshape(-1, initc.shape[1]).cuda()
        initc = initc.expand_as(targc)
        initx = initx.expand(len(targc), -1)
        traj = integrate(vels,initx, targc, initc,ts,kmeans_model,initc[0],targc,config)
        img = decode_imgs(ae, traj[-1]).cpu()
        img = (img.clip(-1, 1) + 1) / 2
        results = torch.cat([x / 255, img], dim=0)
        targc = torch.cat([initc[None, 0], targc], dim=0)
        for j, (img, c) in enumerate(zip(results, targc)):
            img = torch.round(img * 255).cpu().permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            c = "".join(c.cpu().numpy().astype(str))
            if j>0:
                (savedir /c).mkdir(exist_ok=True)
                cv2.imwrite(savedir /c/ f"{i:03}_num_edits_{j}_c_{c}.png", img)
            if j==0:
                (savedir /"original").mkdir(exist_ok=True)
                cv2.imwrite(savedir /"original"/ f"{i:03}_num_edits_{j}_c_{c}.png", img)


    


if __name__ == "__main__":
    validate()
