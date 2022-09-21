import logging
import os
import os.path as osp
import sys
import tempfile
from typing import Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from niloc.data.niloc_datamodule import NilocDataModule
from niloc.network.base_models import ScheduledSamplingModule
from niloc.network.scheduled_1branch import Scheduled1branchModule
from niloc.network.scheduled_2branch import Scheduled2branchModule
from niloc.network.standard_1branch import Standard1branchModule

task_models = {
    "standard_1branch": Standard1branchModule,
    "scheduled_1branch": Scheduled1branchModule,
    "scheduled_2branch": Scheduled2branchModule,
}


def arg_conversion(cfg: DictConfig) -> None:
    """ Convert from time arguments to data size, and append to configuration """

    if not (cfg.data.window.window_time * cfg.data.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (cfg.data.imu_freq / cfg.data.window.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = {
        "window_size": int(cfg.data.window.window_time * cfg.data.imu_freq),
        "step_size": int(cfg.data.imu_freq / cfg.data.window.sample_freq),
        "interval": int(cfg.data.window.window_time * cfg.data.imu_freq),
    }
    net_config = {
        "window_size": data_window_config["window_size"]
    }

    with open_dict(cfg):
        cfg.data_window_cfg = data_window_config
        cfg.net_config = net_config


def assert_and_print(cfg: DictConfig) -> None:
    """Ensure required parameters are present"""

    logging.info(f"Loading data from {cfg.dataset.name} dataset.")
    if cfg.dataset.val_list is None:
        logging.warning("val_list is not specified.")

    # Log the calculated window sizes
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {cfg.data.imu_freq} Hz IMU data")
    logging.info(
        f"Window Size: {cfg.data_window_cfg.window_size}"
    )
    logging.info(
        f"Time: {cfg.data.window.window_time}"
    )


def configure_output(cfg: DictConfig) -> Tuple[str, pl.loggers.TensorBoardLogger]:
    """
    Configure output paths. Find last checkpoint if needed to resume from last checkpoint.
    We use a custom format to enable restarting from last checkpoint automatically.
    returns:
        filepath : output folder for checkpoints
        tb_logger : tensorboard logger
    """

    root_dir = osp.join(cfg.io.root_path, cfg.io.folder_name, cfg.run_name)
    checkpoint_dir, version_str = None, "version_0"
    # check for previous versions
    if osp.exists(root_dir):
        folders = os.listdir(root_dir)
        previous_versions = [
            int(r.rsplit("_", 1)[1])
            for r in folders
            if r.rsplit("_", 1)[0] == "version"
        ]
        if len(previous_versions) > 0:
            last_version = max(previous_versions)
            checkpoint_dir = osp.join(root_dir, f"version_{last_version}")
            version_str = f"version_{last_version + 1}"

    logging.info(f"Identified current run as version {version_str}")

    # setup output directory and logger
    filepath = osp.join(cfg.io.root_path, cfg.io.folder_name, cfg.run_name, version_str)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=root_dir, name="logs", version=version_str)
    logging.info(f"Training output will be written to {filepath}")

    if cfg.train_cfg.resume_from_checkpoint is not None and osp.exists(cfg.train_cfg.resume_from_checkpoint):
        # use specified checkpoint
        logging.info(f"Resuming from checkpoint {cfg.train_cfg.resume_from_checkpoint}")
    elif checkpoint_dir and osp.exists(checkpoint_dir) and cfg.train_cfg.retry_from_last:
        # find last checkpoint if previous versions exists
        checkpoints = os.listdir(checkpoint_dir)
        logging.info(f"{len(checkpoints)}")
        if "last.ckpt" in checkpoints:
            with open_dict(cfg):
                cfg.train_cfg.resume_from_checkpoint = osp.join(
                    checkpoint_dir, "last.ckpt"
                )
        else:
            epochs = [
                int(r.split("epoch=")[1].split(".ckpt")[0].split("-")[0])
                for r in checkpoints
            ]
            if epochs:
                epochs = np.asarray(epochs)
                ckpt_name = checkpoints[np.argmax(epochs)]
                with open_dict(cfg):
                    cfg.train_cfg.resume_from_checkpoint = osp.join(
                        checkpoint_dir, ckpt_name
                    )
        logging.info(f"Resuming from checkpoint {cfg.train_cfg.resume_from_checkpoint}")

    return filepath, tb_logger


def get_model(cfg: DictConfig) -> pl.LightningModule:
    with open_dict(cfg):
        print(f"Overriding output_dim with grid size {cfg.network.output_dim}->{cfg.grid.elements}")
        cfg.network.output_dim = cfg.grid.elements

    model_type = task_models[cfg.task]
    return model_type


def launch_train(cfg: DictConfig) -> None:
    """ Main function for network training with seq->seq categorical output"""
    pl.seed_everything(cfg.random_seed)
    torch.cuda.empty_cache()

    arg_conversion(cfg)
    assert_and_print(cfg)
    model_type = get_model(cfg)

    # Load dataset
    datamodule = NilocDataModule(cfg)
    logging.info("Created dataset.")
    filepath, tb_logger = configure_output(cfg)

    # create network
    if cfg.train_cfg.get("load_weights_only", False):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
            OmegaConf.save(config=cfg, f=fp.name)
            network = model_type.load_from_checkpoint(cfg.train_cfg.load_weights_only,
                                                      hparams_file=fp.name)
        print(f"loaded weights from checkpoint {cfg.train_cfg.load_weights_only}")
        with open_dict(cfg):
            cfg.train_cfg.resume_from_checkpoint = None
            cfg.train_cfg.restore_tr_ratio = cfg.train_cfg.get("restore_tr_ratio", False)  # disable by default
    else:
        network = model_type(cfg)
    logging.info(f'Network {cfg.task} & {cfg.arch.name} loaded: Model type {type(network).__name__}')

    # set model save callbacks
    ckpt_format = "{epoch}-{tr_ratio:.1f}" if issubclass(model_type, ScheduledSamplingModule) else "{epoch}"
    validation_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=filepath,
        monitor="val_loss",
        filename=ckpt_format + "-{val_loss:.2f}",
        save_top_k=10,
    )

    periodic_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=filepath,
        filename=ckpt_format,
        save_top_k=-1,  # save all checkpoints
        period=cfg.train_cfg.get("periodic_save_interval", 10),
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="train_loss_epoch",
        mode="min",
        patience=100000,
    )

    trainer_cfg = {}
    for key in cfg.trainer_cfg:
        trainer_cfg[key] = cfg.trainer_cfg[key]
    logging.info(f"{trainer_cfg}")

    # construct trainer
    trainer = pl.Trainer(
        gpus=cfg.train_cfg.gpus,
        distributed_backend=cfg.train_cfg.accelerator,
        max_epochs=cfg.train_cfg.epochs,
        resume_from_checkpoint=cfg.train_cfg.resume_from_checkpoint,
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            validation_checkpoint_callback,
            periodic_checkpoint_callback,
            early_stopping,
        ],
        plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        **trainer_cfg,
    )

    trainer.fit(network, datamodule=datamodule)
    logging.info("Done")


@hydra.main(config_path="config", config_name="defaults")
def run(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logging.info(OmegaConf.to_yaml(cfg))

    launch_train(cfg)


if __name__ == "__main__":
    run()
