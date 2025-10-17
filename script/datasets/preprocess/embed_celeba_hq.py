import importlib
import torch
from pathlib import Path
import yaml
from omegaconf import DictConfig
import cv2
import numpy as np
from tqdm import tqdm
FIRST_STAGE_MODEL =  "./autoencoder_ckpts/first_stage_models/vq-f4/config.yaml"
IMAGE_DIR = "./datasets/celebADialogHQ/image"
ANNOTATION_TXT = "./datasets/celebADialogHQ/combined_annotation_hq.txt"
OUTPUT_DIR = "./datasets/celebADialogEmbbeded"

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step


if __name__ == "__main__":

    with open(
        FIRST_STAGE_MODEL,
        "rb",
    ) as f:
        config = yaml.safe_load(f)
        config = DictConfig(config)
    model, _ = load_model_from_config(
        config,
        FIRST_STAGE_MODEL,
    )

    image_dir = Path(IMAGE_DIR)
    txt_path = Path(
        ANNOTATION_TXT
    )
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(txt_path, "rt") as f:
        for i, line in enumerate(tqdm(f.readlines())):
            line = line.split()
            if i == 0:
                continue
            x = cv2.imread(image_dir / line[0])
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (256, 256)).reshape(-1, 256, 256, 3).transpose(0, 3, 1, 2)
            c = np.asarray(line[1:]).astype(int)
            img_size = 256
            img_size2 = img_size * img_size
            data_ = x / 127.5 - 1.0
            data_ = torch.from_numpy(data_).cuda()
            torch.cuda.empty_cache()
            with torch.no_grad():
                y = model["model"].encode(data_.float())[0]
            x_emb = y.cpu().numpy()
            np.savez(save_dir / f"batch_{i}", x_emb=x_emb, c=c, x=x)
