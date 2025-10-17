from lightning import LightningModule
from omegaconf import DictConfig
import torch
from hydra.utils import instantiate
from script.datasets.base_dataset import DoubleTerminalDataset
import pdb


class ProcessBasedGenerativeModel(LightningModule):
    """
    Base lightning module for diffusion models and flow matching models
    """

    def __init__(self, myconfig: DictConfig) -> None:
        super().__init__()
        self.config = myconfig
        self.data_handler = instantiate(
            self.config.data_handler,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        self.process_generator = instantiate(
            myconfig.process_generator,
            data_handler=self.data_handler,
            _recursive_=False,
        )
        self.model = instantiate(
            myconfig.model, data_handler=self.data_handler, _recursive_=False
        )
        self.loss = instantiate(myconfig.loss, _recursive_=True)
        self.evaluator = instantiate(
            myconfig.evaluator, model=self, myconfig=myconfig, _recursive_=False
        )
        self.dataset = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim_config = {}
        optim_config["optimizer"] = instantiate(
            self.config.optimizer, params=self.model.parameters()
        )
        if hasattr(self.config, "lr_scheduler"):
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate(
                    self.config.lr_scheduler, optimizer=optim_config["optimizer"]
                )
            }
        return optim_config

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        dataset1 = instantiate(
            self.config.dataset.dataset1,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        dataset2 = instantiate(
            self.config.dataset.dataset2,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        self.dataset = DoubleTerminalDataset(
            self.config.data_config, dataset1, dataset2
        )
        val_dataset1 = instantiate(
            self.config.val_dataset.dataset1,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        val_dataset2 = instantiate(
            self.config.val_dataset.dataset2,
            data_config=self.config.data_config,
            _recursive_=False,
        )
        self.val_dataset = DoubleTerminalDataset(
            self.config.data_config, val_dataset1, val_dataset2
        )

    def forward(self, *args, **kwargs):
        try:
            precision = self.trainer.precision
        except RuntimeError:
            precision = None
        if precision == "16":
            with torch.amp.autocast(dtype=torch.bfloat16):
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)

    def train_dataloader(self):
        return instantiate(self.config.train_dataloader, dataset=self.dataset)

    def val_dataloader(self):
        return instantiate(self.config.val_dataloader, dataset=self.val_dataset)

    def format_batch(self, z0, z1):
        x0, c0 = z0
        x1, c1 = z1
        x0, c0, x1, c1 = (
            x0.reshape(-1, self.config.data_config.x_dim),
            c0.reshape(-1, self.config.data_config.c_dim),
            x1.reshape(-1, self.config.data_config.x_dim),
            c1.reshape(-1, self.config.data_config.c_dim),
        )
        batch_size = min(len(x0), len(x1))
        return x0[:batch_size], c0[:batch_size], x1[:batch_size], c1[:batch_size]

    def validation_step(self, batch, batch_idx):
        metric = self.evaluator.validation(self.format_batch(*batch))
        if metric is not None:
            self.log_dict(metric, logger=True)
        return metric

    def on_train_start(self):
        self.model.train()
        print(self.model)

    def training_step(self, batch, batch_idx):
        x0, c0, x1, c1 = self.format_batch(*batch)
        z1 = self.data_handler.reshape_batch(x1, c1)
        z0 = self.data_handler.reshape_batch(x0, c0)
        self.process_generator.set_batch(z0, z1)
        t = self.process_generator.sample_t()
        u = self.process_generator.construct_target(t)
        zt = self.process_generator.get_model_input(t)
        v = self(zt)
        loss = self.loss(v, u, c=c1)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
