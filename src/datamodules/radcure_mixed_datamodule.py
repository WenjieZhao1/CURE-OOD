import math
import os
import random
from itertools import chain, islice
from typing import Iterable, List, Optional, Sequence, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torchvision.transforms import transforms

from src import utils
from src.datamodules.components.radcure_train_test import RadCureDataset
from src.datamodules.transforms import ExtractPatch, NormalizeIntensity, ToTensor, AddNoise3D


log = utils.get_logger(__name__)


class BalancedMixedBatchSampler(Sampler[List[int]]):
    """Sampler that enforces a fixed ID/OOD ratio inside every batch or per mini-batch group."""

    def __init__(
        self,
        id_indices: Sequence[int],
        ood_indices: Sequence[int],
        batch_size: int,
        id_fraction: float,
        shuffle: bool = True,
        drop_last: bool = False,
        mini_batch_size: Optional[int] = None,
        mini_batches_per_domain: int = 1,
        id_groups: Optional[List[List[int]]] = None,
        ood_groups: Optional[List[List[int]]] = None,
    ) -> None:
        if len(id_indices) == 0 or len(ood_indices) == 0:
            raise ValueError("Both ID and OOD indices are required for balanced sampling")

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.mini_batch_size = mini_batch_size
        self.mini_batches_per_domain = max(1, mini_batches_per_domain)

        self.use_group_mode = bool(mini_batch_size and id_groups and ood_groups)

        if self.use_group_mode:
            self.id_groups = list(id_groups)
            self.ood_groups = list(ood_groups)
            if not self.id_groups or not self.ood_groups:
                raise ValueError("Mini-batch mode requires non-empty ID and OOD group lists")

            self.id_per_batch = mini_batch_size * self.mini_batches_per_domain
            self.ood_per_batch = mini_batch_size * self.mini_batches_per_domain
            self.batch_size = self.id_per_batch + self.ood_per_batch

            self._num_batches = max(
                math.ceil(len(self.id_groups) / self.mini_batches_per_domain),
                math.ceil(len(self.ood_groups) / self.mini_batches_per_domain),
            )
        else:
            if batch_size < 2:
                raise ValueError("batch_size must be >= 2 for mixed sampling")
            if not (0.0 <= id_fraction <= 1.0):
                raise ValueError("id_fraction must be between 0 and 1")

            self.id_indices = list(id_indices)
            self.ood_indices = list(ood_indices)
            self.batch_size = batch_size
            tentative_id = int(round(batch_size * id_fraction))
            self.id_per_batch = min(batch_size - 1, max(1, tentative_id))
            self.ood_per_batch = batch_size - self.id_per_batch
            if self.ood_per_batch == 0:
                self.ood_per_batch = 1
                self.id_per_batch = batch_size - 1

            self._num_batches = max(
                math.ceil(len(self.id_indices) / self.id_per_batch),
                math.ceil(len(self.ood_indices) / self.ood_per_batch),
            )

    def __len__(self) -> int:
        return self._num_batches

    def _cycled_stream(self, base_indices: Sequence) -> Iterable:
        data = list(base_indices)
        while True:
            if self.shuffle:
                random.shuffle(data)
            for idx in data:
                yield idx

    def __iter__(self):
        if self.use_group_mode:
            id_group_stream = self._cycled_stream(self.id_groups)
            ood_group_stream = self._cycled_stream(self.ood_groups)

            for _ in range(self._num_batches):
                id_groups = list(islice(id_group_stream, self.mini_batches_per_domain))
                ood_groups = list(islice(ood_group_stream, self.mini_batches_per_domain))
                batch_groups = id_groups + ood_groups

                if self.shuffle:
                    random.shuffle(batch_groups)

                batch = list(chain.from_iterable(batch_groups))

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                yield batch
        else:
            id_stream = self._cycled_stream(self.id_indices)
            ood_stream = self._cycled_stream(self.ood_indices)

            for _ in range(self._num_batches):
                id_batch = list(islice(id_stream, self.id_per_batch))
                ood_batch = list(islice(ood_stream, self.ood_per_batch))
                batch = id_batch + ood_batch

                if self.shuffle:
                    random.shuffle(batch)

                if self.drop_last and len(batch) < self.batch_size:
                    continue

                yield batch


class RadCureMixedDataModule(LightningDataModule):
    """Lightning datamodule that mixes ID and OOD samples inside every batch."""

    def __init__(
        self,
        root_dir: str,
        id_data_path: str,
        ood_data_path: str,
        train_cache_dir: str,
        test_cache_dir: str,
        train_filename: str = "train.csv",
        batch_size: int = 32,
        id_fraction: float = 0.5,
        ensure_balanced_batches: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
        pin_memory: bool = True,
        patch_xy: int = 80,
        patch_z: int = 48,
        patch_sz: int = 8,
        p_tumor: float = 0.5,
        time_bins_data: int = 10,
        patient_split: bool = True,
        id_domain_label: int = 0,
        ood_domain_label: int = 1,
        mini_batch_size: Optional[int] = None,
        mini_batches_per_domain: int = 1,
        apply_noise: bool = False,
        noise_type: str = "gaussian",
        noise_severity: int = 1,
        noise_probability: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.id_fraction = id_fraction
        self.ensure_balanced_batches = ensure_balanced_batches
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.mini_batch_size = mini_batch_size
        self.mini_batches_per_domain = mini_batches_per_domain
        self._noise_enabled = apply_noise
        self._noise_type = noise_type
        self._noise_severity = noise_severity
        self._noise_probability = noise_probability

        test_transform_steps = [
            NormalizeIntensity(),
            ExtractPatch(
                patch_size=[patch_z, patch_xy, patch_xy],
                p_tumor=1,
            ),
        ]

        if self._noise_enabled:
            test_transform_steps.append(
                AddNoise3D(
                    noise_type=self._noise_type,
                    severity=self._noise_severity,
                    probability=self._noise_probability,
                )
            )

        test_transform_steps.append(ToTensor())

        self.test_transforms = transforms.Compose(test_transform_steps)

        if self._noise_enabled:
            log.info(
                "Noise augmentation enabled for testing: type=%s severity=%d probability=%.2f",
                self._noise_type,
                self._noise_severity,
                self._noise_probability,
            )

        self._id_dataset: Optional[RadCureDataset] = None
        self._ood_dataset: Optional[RadCureDataset] = None
        self.data_test: Optional[ConcatDataset] = None
        self._id_groups_local: Optional[List[List[int]]] = None
        self._ood_groups_local: Optional[List[List[int]]] = None
        self._id_groups_global: Optional[List[List[int]]] = None
        self._ood_groups_global: Optional[List[List[int]]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "test"):
            return

        root_dir = self.hparams["root_dir"]
        test_cache_dir = self.hparams["test_cache_dir"]
        num_workers = self.hparams["num_workers"]
        time_bins = self.hparams["time_bins_data"]
        patient_split = self.hparams["patient_split"]

        id_path = self._resolve_path(self.hparams["id_data_path"], root_dir)
        ood_path = self._resolve_path(self.hparams["ood_data_path"], root_dir)

        log.info("Loading ID data from %s", id_path)
        self._id_dataset = RadCureDataset(
            root_directory=root_dir,
            clinical_data_path=id_path,
            train_mode=False,
            patch_size=self.hparams["patch_sz"],
            time_bins=time_bins,
            cache_dir=test_cache_dir,
            transform=self.test_transforms,
            num_workers=num_workers,
            num_classes=self.hparams.get("num_classes", 2),
            patient_split=patient_split,
            train_filename=self.hparams["train_filename"],
            domain_label=self.hparams["id_domain_label"],
        )

        log.info("Loading OOD data from %s", ood_path)
        self._ood_dataset = RadCureDataset(
            root_directory=root_dir,
            clinical_data_path=ood_path,
            train_mode=False,
            patch_size=self.hparams["patch_sz"],
            time_bins=time_bins,
            cache_dir=test_cache_dir,
            transform=self.test_transforms,
            num_workers=num_workers,
            num_classes=self.hparams.get("num_classes", 2),
            patient_split=patient_split,
            train_filename=self.hparams["train_filename"],
            domain_label=self.hparams["ood_domain_label"],
        )

        if self.mini_batch_size:
            (
                id_assignments,
                self._id_groups_local,
                id_trimmed,
            ) = self._build_group_assignments(len(self._id_dataset), self.mini_batch_size, start_group_id=0)
            (
                ood_assignments,
                self._ood_groups_local,
                ood_trimmed,
            ) = self._build_group_assignments(
                len(self._ood_dataset),
                self.mini_batch_size,
                start_group_id=len(self._id_groups_local or []),
            )

            self._id_dataset.mini_batch_assignments = id_assignments
            self._ood_dataset.mini_batch_assignments = ood_assignments

            if self._id_groups_local:
                self._id_groups_global = [group[:] for group in self._id_groups_local]
            if self._ood_groups_local:
                offset = len(self._id_dataset)
                self._ood_groups_global = [[idx + offset for idx in group] for group in self._ood_groups_local]

            if id_trimmed > 0:
                log.warning(
                    "Dropped %d ID samples that do not fit into mini-batches of size %d",
                    id_trimmed,
                    self.mini_batch_size,
                )
            if ood_trimmed > 0:
                log.warning(
                    "Dropped %d OOD samples that do not fit into mini-batches of size %d",
                    ood_trimmed,
                    self.mini_batch_size,
                )

        self.data_test = ConcatDataset([self._id_dataset, self._ood_dataset])

        log.info(
            "Created mixed test dataset with %d ID and %d OOD samples",
            len(self._id_dataset),
            len(self._ood_dataset),
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise RuntimeError("DataModule.setup('test') must be called before test_dataloader()")

        if (
            self.ensure_balanced_batches
            and len(self._id_dataset) > 0
            and len(self._ood_dataset) > 0
        ):
            effective_batch_size = self.batch_size
            effective_fraction = self.id_fraction
            if self.mini_batch_size:
                effective_batch_size = self.mini_batch_size * self.mini_batches_per_domain * 2
                effective_fraction = 0.5

            sampler = BalancedMixedBatchSampler(
                id_indices=range(len(self._id_dataset)),
                ood_indices=range(
                    len(self._id_dataset),
                    len(self._id_dataset) + len(self._ood_dataset),
                ),
                batch_size=effective_batch_size,
                id_fraction=effective_fraction,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                mini_batch_size=self.mini_batch_size,
                mini_batches_per_domain=self.mini_batches_per_domain,
                id_groups=self._id_groups_global,
                ood_groups=self._ood_groups_global,
            )

            return DataLoader(
                dataset=self.data_test,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        log.warning("Falling back to standard sequential sampling for mixed testing")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def _resolve_path(path: str, root_dir: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(root_dir, path)

    @staticmethod
    def _build_group_assignments(
        num_samples: int,
        mini_batch_size: int,
        start_group_id: int = 0,
    ) -> Tuple[List[Optional[int]], List[List[int]], int]:
        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be a positive integer")

        assignments: List[Optional[int]] = [None for _ in range(num_samples)]
        groups: List[List[int]] = []

        full_groups = num_samples // mini_batch_size
        usable_samples = full_groups * mini_batch_size

        for group_idx in range(full_groups):
            indices = list(range(group_idx * mini_batch_size, (group_idx + 1) * mini_batch_size))
            groups.append(indices)
            global_group_id = start_group_id + group_idx
            for sample_idx in indices:
                assignments[sample_idx] = global_group_id

        trimmed = num_samples - usable_samples
        return assignments, groups, trimmed
