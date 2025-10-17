from script.evaluator.base_evaluator import BaseEvaluator
import numpy as np
import torch
from matplotlib import pyplot as plt
from lightning.pytorch import loggers as pl_loggers
from torchvision.utils import make_grid, save_image
import ot
from omegaconf import DictConfig
import yaml
from script.datasets.preprocess.embed_celeba_hq import load_model_from_config
from script.datasets.CelebAHQ_dataset import CLASS_NAMES
from pathlib import Path
from tqdm.auto import tqdm
import math
import cv2


class CelebAHQEvaluator(BaseEvaluator):

    def __init__(
        self,
        ae_config: str,
        ae_ckpt: str,
        targc: list,
        img_path: str = "./",
        initial_path: str = "./datasets/CelebaEmbedded_eval",
        c_indecies: list = [0, 1, 2, 3, 4],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = 3
        self.image_size = 256
        self.emb_size = 64
        self.ae_config = ae_config
        self.ae_ckpt = ae_ckpt
        self.targc = targc
        self.img_path = img_path
        self.initial_path = initial_path
        self.c_indecies = c_indecies

    def load_image(self, path):
        data = np.load(path)
        x_emb = (
            torch.from_numpy(data["x_emb"])
            .reshape(1, self.in_channels * self.emb_size * self.emb_size)
            .cuda()
        )
        x = torch.from_numpy(data["x"])
        c = torch.from_numpy(data["c"][self.c_indecies]).reshape(1, len(self.c_indecies)).cuda()
        return x, c, x_emb

    def evaluate(self):
        initial_files = list(Path(self.initial_path).glob("*.npz"))
        savedir = Path(self.img_path)
        savedir.mkdir(exist_ok=True, parents=True)
        self.prepare_ae()
        print(CLASS_NAMES)
        for i, img_path in enumerate(tqdm(initial_files)):
            x, initc, initx = self.load_image(img_path)
            targc = torch.tensor(self.targc).reshape(-1, initc.shape[1]).cuda()
            initc = initc.expand_as(targc)
            initx = initx.expand(len(targc), -1)
            traj = self.integrate(initx, targc, initc)
            img = self.decode_imgs(traj[-1]).cpu()
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

    def prepare_ae(self):
        with open(
            self.ae_config,
            "rb",
        ) as f:
            config = yaml.safe_load(f)
            config = DictConfig(config)
        model, _ = load_model_from_config(
            config,
            self.ae_ckpt,
        )
        self.ae = model

    def validation(self, batch: torch.Tensor) -> dict:
        initx, initc, targx, targc = batch
        self.prepare_ae()
        print(CLASS_NAMES)
        print(f"starting points: {initc}")
        print(f"final points {targc}")
        traj = self.integrate(initx, targc, initc)
        traj = self.decode_traj(traj).cpu()
        traj = (traj.clip(-1, 1) + 1) / 2
        grid = self.plot_image_traj(traj[:, :10])
        self.log_image(grid, "trans-image-direct")
        traj = self.integrate_ct(initx, targc, initc)
        traj = self.decode_traj(traj).cpu()
        traj = (traj.clip(-1, 1) + 1) / 2
        grid = self.plot_image_traj(traj[:, :10])
        self.log_image(grid, "trans-image-ct")

    @torch.no_grad()
    def decode_traj(self, traj: torch.Tensor) -> torch.Tensor:
        T, b, d = traj.shape
        decoder = self.ae["model"].to(traj.device)
        x = traj.reshape(T * b, self.in_channels, self.emb_size, self.emb_size)
        batch_size = 16
        img = torch.zeros(T * b, self.in_channels, self.image_size, self.image_size)
        for i in range(0, len(x), batch_size):
            img[i : (i + batch_size)] = decoder.decode(x[i : (i + batch_size)])
        img[i:] = decoder.decode(x[i:])
        traj = img.reshape(T, b, self.in_channels, self.image_size, self.image_size)
        return traj

    @torch.no_grad()
    def decode_imgs(self, latent: torch.Tensor) -> torch.Tensor:
        b, d = latent.shape
        decoder = self.ae["model"].to(latent.device)
        x = latent.reshape(b, self.in_channels, self.emb_size, self.emb_size)
        batch_size = 16
        img = torch.zeros(b, self.in_channels, self.image_size, self.image_size)
        for i in range(0, len(x), batch_size):
            img[i : (i + batch_size)] = decoder.decode(x[i : (i + batch_size)])
        img[i:] = decoder.decode(x[i:])
        img = img.reshape(b, self.in_channels, self.image_size, self.image_size)
        return img

    def plot_image_traj(self, traj: torch.Tensor) -> plt.Figure:
        t, b, _, _, _ = traj.shape
        grid = make_grid(
            traj.permute(1, 0, 2, 3, 4).reshape(
                t * b, self.in_channels, self.image_size, self.image_size
            ),
            nrow=t,
        )
        return grid

    def log_image(self, image, logname):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.model.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        tb_logger.add_image(logname, image, self.model.global_step)

    def plt_to_npy(self, fig):
        fig.canvas.draw()  # Canvasに描画
        plot_image = fig.canvas.renderer._renderer  # プロットを画像データとして取得
        # tensorboardXはchannel firstのようなので、それに合わせる
        plot_image_array = np.array(plot_image).transpose(2, 0, 1)
        return plot_image_array


class CelebAHQCFGDiffusionEvaluator(CelebAHQEvaluator):
    def __init__(
        self,
        guidance_weight: list = [
            0.3,
        ],
        timesteps_eval: float = 0.5,
        guidance_weight_eval: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.guidance_weight = guidance_weight
        self.guidance_weight_eval = guidance_weight_eval
        self.timesteps_eval = timesteps_eval

    def add_noise(self, x0: torch.Tensor, t: torch.tensor):
        b = math.atan(math.exp(-self.data_handler.lam_max * 0.5))
        a = math.atan(math.exp(-self.data_handler.lam_min * 0.5)) - b
        u = -2 * torch.log(torch.tan(a * t + b))
        noise = torch.randn_like(x0)
        zt = (
            self.data_handler.alpha_sqs(u) * x0 + self.data_handler.sigma_sqs(u) * noise
        )
        return zt

    def evaluate(self):
        initial_files = list(Path(self.initial_path).glob("*.npz"))
        savedir = Path(self.img_path)
        savedir.mkdir(exist_ok=True, parents=True)
        self.prepare_ae()
        print(CLASS_NAMES)
        for i, img_path in enumerate(tqdm(initial_files)):
            x, initc, initx = self.load_image(img_path)
            targc = torch.tensor(self.targc).reshape(-1, initc.shape[1]).cuda()
            initc = initc.expand_as(targc)
            initx = initx.expand(len(targc), -1)
            traj = self.partial_diffusion(
                initx, targc, w=self.guidance_weight_eval, timesteps=self.timesteps_eval
            )
            img = self.decode_imgs(traj[-1]).cpu()
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

    def validation(self, batch: torch.Tensor) -> dict:
        initx, initc, targx, targc = batch
        self.prepare_ae()
        print(CLASS_NAMES)
        print(f"starting points: {initc}")
        print(f"final points {targc}")
        for w in tqdm(self.guidance_weight):
            traj = self.cfg_diffusion_integrate(initx, targc, w=w)
            traj = self.decode_traj(traj).cpu()
            traj = (traj.clip(-1, 1) + 1) / 2
            T = len(traj)
            grid = self.plot_image_traj(traj[:: T // 10, :10])
            self.log_image(grid, f"trans-image-{w}")


class CelebAHQCFGFlowEvaluator(CelebAHQEvaluator):
    def __init__(self, guidance_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.guidance_weight = guidance_weight

    def evaluate(self):
        x, initc, initx = self.load_image()
        targc = torch.tensor(self.targc).reshape(-1, 5).cuda()
        initc = initc.expand_as(targc)
        initx = initx.expand(len(targc), -1)
        self.prepare_ae()
        print(CLASS_NAMES)
        print(f"starting points: {initc}")
        print(f"final points {targc}")
        traj = self.cfg_flow_integrate(initx, targc, w=self.guidance_weight)
        traj = self.decode_traj(traj).cpu()
        traj = (traj.clip(-1, 1) + 1) / 2
        T = len(traj)
        grid = self.plot_image_traj(traj[:: T // 10, :10])
        with open(Path(self.img_path) / "generated_images.jpg", "wb") as f:
            save_image(grid, f)
        traj = self.cfg_flow_nocond_integrate(initx, targc)
        traj = self.decode_traj(traj).cpu()
        traj = (traj.clip(-1, 1) + 1) / 2
        T = len(traj)
        grid = self.plot_image_traj(traj[:: T // 10, :10])
        with open(Path(self.img_path) / "generated_nocond_images.jpg", "wb") as f:
            save_image(grid, f)

    def validation(self, batch: torch.Tensor) -> dict:
        initx, initc, targx, targc = batch
        self.prepare_ae()
        print(CLASS_NAMES)
        print(f"starting points: {initc}")
        print(f"final points {targc}")
        for w in tqdm(self.guidance_weight):
            traj = self.cfg_flow_integrate(initx, targc, w=w)
            traj = self.decode_traj(traj).cpu()
            traj = (traj.clip(-1, 1) + 1) / 2
            T = len(traj)
            grid = self.plot_image_traj(traj[:: T // 10, :10])
            self.log_image(grid, f"trans-image-{w}")
        traj = self.cfg_flow_nocond_integrate(initx, targc)
        traj = self.decode_traj(traj).cpu()
        traj = (traj.clip(-1, 1) + 1) / 2
        T = len(traj)
        grid = self.plot_image_traj(traj[:: T // 10, :10])
        self.log_image(grid, "no-cond-image")
