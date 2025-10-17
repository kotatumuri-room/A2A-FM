from hydra.utils import instantiate
import lightning as L
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from script.process_based_generative_model import ProcessBasedGenerativeModel


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def train(config: DictConfig):
    configpath = Path(f"{config.savedir}/config.yaml")
    configpath.parent.mkdir(parents=True, exist_ok=True)
    if config.get("seed"):
        print(f"fixing seed to {config.seed}")
        L.seed_everything(config.seed, workers=True)
    model: ProcessBasedGenerativeModel = ProcessBasedGenerativeModel(config)
    logger: List[Logger] = instantiate(config.logger, _recursive_=False)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=1000,verbose=False)
    trainer = instantiate(
        config.trainer, logger=logger, default_root_dir=config.savedir, callbacks=[ckpt_callback]
    )

    ckpt_path = None
    if hasattr(config, "ckpt_dir"):
        if config.ckpt_dir is not None:
            ckpt_dir = Path(config.ckpt_dir)
            ckpt_path = list(ckpt_dir.glob("*.ckpt"))[0]
            print(f"using check point {ckpt_path}")
    print(trainer)
    trainer.fit(model=model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    train()
