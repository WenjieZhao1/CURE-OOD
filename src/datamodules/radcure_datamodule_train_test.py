from typing import Optional, Tuple
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.datamodules.transforms import *

from src.datamodules.components.radcure_train_test import RadCureDataset
import pandas as pd

class RadCureDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = self.hparams["data_dir"]
      
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.pin_memory = self.hparams["pin_memory"]
        self.Fold = self.hparams["Fold"]
        self.p_tumor = self.hparams["p_tumor"]

        self.test_transforms = transforms.Compose(
            [ 
             NormalizeIntensity(),
             ExtractPatch(patch_size=[self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]], p_tumor=1),
             ToTensor()]
        )

        self.train_transforms = transforms.Compose([
            NormalizeIntensity(),
            RandomRotation(),
            ExtractPatch(patch_size=[self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]], p_tumor=self.p_tumor),
            ToTensor(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _clinical_data_root(self) -> str:
        data_dir = self.hparams.get("data_dir")
        if data_dir and os.path.exists(data_dir):
            return data_dir

        fallback = os.path.join(self.hparams["root_dir"], "data")
        if os.path.exists(fallback):
            return fallback

        home_fallback = os.path.join(os.path.expanduser("~"), "data")
        if os.path.exists(home_fallback):
            return home_fallback

        return data_dir

    def _resolve_csv_path(self, filename: Optional[str]) -> Optional[str]:
        if not filename:
            return None
        if os.path.isabs(filename):
            return filename
        data_root = self._clinical_data_root()
        if data_root and data_root.endswith(".csv"):
            data_root = os.path.dirname(data_root)
        return os.path.join(data_root, filename)


    def setup(self, stage: Optional[str] = None):
        train_filename = self.hparams.get("train_filename", "RADCURE_Training.csv")
        test_filename = self.hparams.get("test_filename", "RADCURE_Testing.csv")
        train_path = self._resolve_csv_path(train_filename)
        test_path = self._resolve_csv_path(test_filename)
        clinical_data_root = self._clinical_data_root()

        if stage == "test":
            clinical_data_path = test_path if test_path else clinical_data_root
            self.data_test = RadCureDataset(self.hparams["root_dir"],
                                clinical_data_path,
                                False, # testing
                                self.hparams["patch_sz"],
                                self.hparams["time_bins_data"],
                                transform=self.test_transforms,
                                cache_dir=self.hparams["test_cache_dir"],
                                num_workers=self.hparams["num_workers"],
                                train_filename=train_filename,
                                test_filename=test_filename)
        else:
            training_data_path = train_path if train_path else clinical_data_root
            train_dataset_full = RadCureDataset(self.hparams["root_dir"],
                                      training_data_path,
                                      True, # training
                                      self.hparams["patch_sz"],
                                      self.hparams["time_bins_data"],
                                      transform=self.train_transforms,
                                      cache_dir=self.hparams["train_cache_dir"],
                                      num_workers=self.hparams["num_workers"],
                                      train_filename=train_filename,
                                      test_filename=test_filename)
            val_dataset_full = RadCureDataset(self.hparams["root_dir"],
                                      training_data_path,
                                      True, # validation is a held-out fold of the training CSV
                                      self.hparams["patch_sz"],
                                      self.hparams["time_bins_data"],
                                      transform=self.test_transforms,
                                      cache_dir=self.hparams["train_cache_dir"],
                                      num_workers=self.hparams["num_workers"],
                                      train_filename=train_filename,
                                      test_filename=test_filename)

            if train_path:
                split_df_path = train_path
            elif clinical_data_root and clinical_data_root.endswith(".csv"):
                split_df_path = clinical_data_root
            else:
                split_df_path = os.path.join(clinical_data_root, train_filename)
            df = pd.read_csv(split_df_path)

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5820222)

            train_idx = {}
            test_idx = {}

            key = 1
            for i,j in kf.split(df,df['RF']):
                train_idx[key] = i
                test_idx[key] = j

                key += 1

            self.data_train = Subset(train_dataset_full, train_idx[self.Fold])
            self.data_val = Subset(val_dataset_full, test_idx[self.Fold])


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
