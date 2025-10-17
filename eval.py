from hydra.utils import instantiate
import lightning as L
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from script.process_based_generative_model import ProcessBasedGenerativeModel
from script.evaluator.base_evaluator import BaseEvaluator


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def evaluate(config: DictConfig):
    configpath = Path(f"{config.savedir}/config.yaml")
    configpath.parent.mkdir(parents=True, exist_ok=True)
    if config.get("seed"):
        print(f"fixing seed to {config.seed}")
        L.seed_everything(config.seed, workers=True)

    ckpt_dir = Path(config.ckpt_dir)
    ckpt_path = list(ckpt_dir.glob("*.ckpt"))[0]
    print(f"using ckpt: {ckpt_path.stem}")
    model: ProcessBasedGenerativeModel = (
        ProcessBasedGenerativeModel.load_from_checkpoint(ckpt_path, myconfig=config)
    )
    evaluator = instantiate(
        config.evaluator, model=model, myconfig=config, _recursive_=False
    )
    evaluator.evaluate()


if __name__ == "__main__":
    evaluate()
