import logging
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from niloc.data.dataset_velocity_reloc import (
    GlobalLocDataset,
    VelocityGridSequence,
)

from niloc.data.transforms import (
    ComposeTransform,
    PerturbScale,
    ReverseDirection,
    RotateYawFeat,
    RotateYawFeatManhatten,
    BiasShift,
    YawDrift,
)

dataset_classes = {
    "GlobalLocDataset": GlobalLocDataset,
}
sequence_classes = {
    "VelocityGridSequence": VelocityGridSequence,
}


class NilocDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module to be used with any dataset class.
    """

    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(
            self,
            cfg: DictConfig,
            mode: str = "train",
    ) -> None:
        """
        Args:
            - cfg - configuration
            - sequence_class - reference to ProcessedSequence class
            - mode - [ train, val, test]
        """
        super().__init__()
        self._cfg = cfg
        self._sequence_class = sequence_classes[cfg.data.classes.sequence]
        self._mode = mode

        # assign dataset class
        self._data_class = dataset_classes[cfg.data.classes.dataset]

    def _create_transforms(self, mode: str) -> ComposeTransform:
        """Create transforms according to cfg and mode"""
        transforms = []
        if mode in ["train", "val", "eval"]:
            if self._cfg.data.transformers.rotate_yaw == "manhatten":
                transforms.append(RotateYawFeatManhatten())
            elif self._cfg.data.transformers.rotate_yaw == "random":
                transforms.append(RotateYawFeat())
        if mode == "train":
            if self._cfg.data.transformers.get('do_bias_shift', False): # for IMU data
                transforms.append(
                    BiasShift(
                        self._cfg.data.transform.gyro_bias_range,
                        self._cfg.data.transform.accel_bias_range,
                    )
                )
            if self._cfg.data.transformers.get("yaw_drift", False):
                transforms.append(
                    YawDrift(self._cfg.data.transform.yaw_drift_sigma)
                )
            if self._cfg.data.transformers.get("perturb_scale", False):
                transforms.append(
                    PerturbScale(self._cfg.data.transform.scale_sigma)
                )
            if self._cfg.data.transformers.get("reverse_direction", False):
                transforms.append(ReverseDirection())

        return ComposeTransform(transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create dataset.
        """
        self.train_dataset = self._data_class(
            self._sequence_class,
            self._cfg,
            self._cfg.dataset.root_dir,
            data_list_file=self._cfg.dataset.train_list,
            mode="train",
            transform=self._create_transforms("train"),
        )
        logging.info(f"Train dataset loaded. Length: {len(self.train_dataset)}")

        self.val_dataset = self._data_class(
            self._sequence_class,
            self._cfg,
            self._cfg.dataset.root_dir,
            data_list_file=self._cfg.dataset.val_list,
            mode="val",
            transform=self._create_transforms("val"),
        )
        logging.info(f"Validation dataset loaded. Length: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self._cfg.data.batch_size,
            num_workers=self._cfg.train_cfg.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self._cfg.data.batch_size,
            num_workers=self._cfg.train_cfg.num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
