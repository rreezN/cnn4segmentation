import numpy as np
import torch
from model import UNerveV1, UNerveV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from dataloader import MyDataModule
import wandb
import random
import yaml
from typing import Dict, Any
from pytorch_lightning.loggers import WandbLogger

PARAMS = {
    "model_name": "UNerveV1",
    "project_name": "Wet-Bone-rrr",
    "seed": 420,
    "n_channels": 1,
    "n_classes": 2,
    "num_epochs": 150,
    "patience": 30,
    "learning_rate": 1e-3,
    "dropout": 0.3,
    "batch_dict": {0: 4},
    "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
    "limit_train_batches": 0.2,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "activation_function": "ReLU",
    "data_size": "508x508"
}

random.seed(PARAMS["seed"])
torch.manual_seed(PARAMS["seed"])
np.random.seed(PARAMS["seed"])


def train() -> None:
    if PARAMS["model_name"] == "UNerveV1":
        model = UNerveV1(PARAMS)
    elif PARAMS["model_name"] == "UNerveV2":
        model = UNerveV2(PARAMS)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/" + PARAMS["model_name"],
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=PARAMS["patience"],
        verbose=True,
        mode="min"
    )

    wandb_logger = WandbLogger(
        project=PARAMS["project_name"], entity="nerve-poster", log_model="all"
    )

    for key, val in PARAMS.items():
        if key == 'batch_dict':
            wandb_logger.experiment.config[key] = [(key_, val_) for key_, val_ in val.items()]
        else:
            wandb_logger.experiment.config[key] = val

    trainer = Trainer(
        devices="auto",
        accelerator=PARAMS["accelerator"],
        max_epochs=PARAMS["num_epochs"],
        limit_train_batches=PARAMS["limit_train_batches"],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback, TQDMProgressBar(refresh_rate=1)],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger
    )

    data_loader = MyDataModule(batch_dict=PARAMS["batch_dict"],
                               device=PARAMS["accelerator"],
                               data_size=PARAMS["data_size"])

    trainer.fit(model, datamodule=data_loader)
    trainer.test(model, datamodule=data_loader)


def train_sweep() -> None:
    with open('./sweep.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=sweep_config, project=PARAMS["project_name"], entity="audiobots")

    def sweep_run() -> None:
        with wandb.init() as run:
            hparams = {**run.config}  # **DEFAULT_PARAMS,
            train(hparams)

    wandb.agent(sweep_id=sweep_id, function=sweep_run, count=None)


if __name__ == "__main__":
    # train_sweep()
    # hparams = {"learning_rate": 1e-3,
    #            "optimizer": "adam",
    #            "loss_function": "cross_entropy",
    #            "activation_function": "ReLU",
    #            "dropout": 0.3}
    # train(hparams)
    train()
