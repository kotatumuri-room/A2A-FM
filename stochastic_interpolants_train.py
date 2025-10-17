import hydra
from hydra.utils import instantiate
from stochastic_interpolant.vector_field import VelocityField
from stochastic_interpolant.path_optimizer import PathOptimizer
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from itertools import combinations


def train(model, config, ckpt_dir,retrain=True,max_steps_spec=None):
    logger = instantiate(config.logger, _recursive_=False,save_dir=ckpt_dir)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=10, verbose=True)
    max_epochs = min(config.trainer.max_epochs, 1000)
    max_steps = -1
    if max_steps_spec is not None:
        max_steps = min(max_epochs,max_steps_spec)
    trainer = instantiate(
        config.trainer,
        logger=logger,
        default_root_dir=ckpt_dir,
        callbacks=[ckpt_callback],
        max_epochs=max_epochs,
        max_steps = max_steps
    )
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = None
    if ckpt_dir.exists() and not retrain:
        ckpt_path = list(ckpt_dir.glob("*.ckpt"))[0]
        print(f"using check point {ckpt_path}")
    trainer.fit(model=model, ckpt_path=ckpt_path)


@hydra.main(
    config_path="stochastic_interpolants_config", config_name="stochastic_interpolants", version_base="1.2"
)
def main(config):
    datasets = instantiate(config.datasets, _recursive_=False)
    K = len(datasets)
    
    root_dir = Path(config.savedir)
    ckpt_dirs_vf = [root_dir / f"g{k}" for k in range(K)]
    if hasattr(config,"ckpt_versions_gt"):
        ckpt_paths = [list((ckpt_dir/"lightning_logs"/f"version_{v}"/"checkpoints").glob("*.ckpt"))[0] for ckpt_dir, v in zip(ckpt_dirs_vf,config.ckpt_versions_gt)]
        print(ckpt_paths)
        velocity_fields = [VelocityField.load_from_checkpoint(ckpt_path,myconfig=config.vector_field) for ckpt_path in ckpt_paths]
    else:
        velocity_fields = [
        VelocityField(datasets, myconfig=config.vector_field, k=k) for k in range(K)
        ]
        for vf, ckpt_dir in zip(velocity_fields, ckpt_dirs_vf):
            train(vf, config, ckpt_dir)
    for i, j in combinations(range(K), 2):
        ckpt_dir = root_dir / f"alpha_{i}_{j}"
        alpha = PathOptimizer(velocity_fields, datasets, config.alpha, i, j)
        train(alpha, config, ckpt_dir,max_steps_spec=500)


if __name__=="__main__":
    main()
