import os
from typing import Callable, List, Optional, Tuple
import sys

import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import nibabel as nib
import pathlib
from einops import rearrange

from joblib import Parallel, delayed

from sklearn.preprocessing import scale

from torchmtlr.utils import make_time_bins, encode_survival

from src import utils

log = utils.get_logger(__name__)

def get_paths_to_patient_files(train_mode, path_to_imgs, PatientID, append_mask=True):
    path_to_imgs = pathlib.Path(path_to_imgs)

    patients = [p for p in PatientID] # if os.path.isdir(path_to_imgs / p)
    paths = []
    # Patient image and mask paths are derived from each patient ID.
    for p in patients:
        path_to_ct = path_to_imgs / (p + '_image.nii.gz')

        if append_mask:
            if train_mode:
                path_to_mask = path_to_imgs/ (p + '_mask_GTV.nii.gz')
            else:
                path_to_mask = path_to_imgs/ (p + '_mask_GTV.nii.gz')
            paths.append((path_to_ct, path_to_mask))
        else:
            paths.append((path_to_ct))
    return paths

class RadCureDataset(Dataset):
    LABEL_COLUMNS = ['ID', 'time1', 'event1', 'time2', 'event2', 'time3', 'event3', 'time4', 'event4']

    def __init__(self,
                 root_directory:str, 
                 clinical_data_path:str, 
                 train_mode:bool = False,
                 patch_size:int =50,
                 time_bins:int = 14,
                 cache_dir:str = "data_cropped/data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1,
                 num_classes: int = 2,
                 patient_split:bool =True,
                 train_filename: str = "RADCURE_Training.csv",
                 test_filename: str = "RADCURE_Testing.csv",
                 domain_label: Optional[int] = None,
                 mini_batch_assignments: Optional[List[int]] = None
    ):
        requested_cache_dir = cache_dir
        self.num_of_seqs = 1 #CT only

        self.train_mode = train_mode
        
        self.root_directory = root_directory
        self.patch_size = patch_size

        self.transforms = transform
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.clinical_feature_columns = self._clinical_feature_columns(clinical_data_path)
        self.clinical_data = self.make_data(clinical_data_path)
        cache_dir = self._resolve_cache_dir(requested_cache_dir, self.clinical_data['ID'])
        self.domain_label = domain_label
        self.mini_batch_assignments = mini_batch_assignments
        if self.train_mode: 
            if patient_split:
                self.time_bins1 = make_time_bins(times=self.clinical_data["time1"], num_bins=time_bins)
                self.time_bins2 = make_time_bins(times=self.clinical_data["time2"], num_bins=time_bins)
                self.time_bins3 = make_time_bins(times=self.clinical_data["time3"], num_bins=time_bins)
                self.time_bins4 = make_time_bins(times=self.clinical_data["time4"], num_bins=time_bins)
            else:
                self.time_bins1 = make_time_bins(times=self.clinical_data["time1"], num_bins=time_bins, event = self.clinical_data["event1"])
                self.time_bins2 = make_time_bins(times=self.clinical_data["time2"], num_bins=time_bins, event = self.clinical_data["event2"])
                self.time_bins3 = make_time_bins(times=self.clinical_data["time3"], num_bins=time_bins, event = self.clinical_data["event3"])
                self.time_bins4 = make_time_bins(times=self.clinical_data["time4"], num_bins=time_bins, event = self.clinical_data["event4"])


        else:
            self.time_bins1 = self.load_training_time_bines(patient_split, clinical_data_path, "event1", "time1", time_bins) 
            self.time_bins2 = self.load_training_time_bines(patient_split, clinical_data_path, "event2", "time2", time_bins) 
            self.time_bins3 = self.load_training_time_bines(patient_split, clinical_data_path, "event3", "time3", time_bins) 
            self.time_bins4 = self.load_training_time_bines(patient_split, clinical_data_path, "event4", "time4", time_bins) 

        log.info(self.time_bins1.cpu().numpy())
        log.info(self.time_bins2.cpu().numpy())
        log.info(self.time_bins3.cpu().numpy())
        log.info(self.time_bins4.cpu().numpy())
        self.y1 = encode_survival(self.clinical_data["time1"].values, self.clinical_data["event1"].values, self.time_bins1) # single event
        self.y2 = encode_survival(self.clinical_data["time2"].values, self.clinical_data["event2"].values, self.time_bins2) # single event
        self.y3 = encode_survival(self.clinical_data["time3"].values, self.clinical_data["event3"].values, self.time_bins3) # single event
        self.y4 = encode_survival(self.clinical_data["time4"].values, self.clinical_data["event4"].values, self.time_bins4) # single event
        self.cache_path = get_paths_to_patient_files(self.train_mode, cache_dir, self.clinical_data['ID'])

    def _fallback_clinical_dirs(self):
        candidates = [os.path.join(self.root_directory, 'data')]
        env_dir = os.environ.get("RADCURE_CSV_DIR")
        if env_dir:
            candidates.append(env_dir)
        return candidates

    @staticmethod
    def _unique_existing_path_candidates(paths):
        seen = set()
        for path in paths:
            if not path:
                continue
            path = str(path)
            if path in seen:
                continue
            seen.add(path)
            yield path

    def _resolve_data_csv(self, path):
        filename = self.train_filename if self.train_mode else self.test_filename
        candidates = []

        if path:
            path = str(path)
            if path.endswith('.csv'):
                candidates.append(path)
                for base_dir in self._fallback_clinical_dirs():
                    candidates.append(os.path.join(base_dir, os.path.basename(path)))
            else:
                candidates.append(os.path.join(path, filename))

        for base_dir in self._fallback_clinical_dirs():
            candidates.append(os.path.join(base_dir, filename))

        tried = list(self._unique_existing_path_candidates(candidates))
        for candidate in tried:
            if os.path.exists(candidate):
                if path and candidate != path and not str(path).endswith('.csv'):
                    log.info(f"Using fallback clinical CSV: {candidate}")
                elif path and str(path).endswith('.csv') and candidate != path:
                    log.info(f"Using fallback clinical CSV: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Unable to locate {filename}. Tried: {', '.join(tried)}"
        )

    def _resolve_cache_dir(self, cache_dir, patient_ids):
        candidates = [cache_dir, os.environ.get("RADCURE_NIFTI_DIR")]
        tried = list(self._unique_existing_path_candidates(candidates))

        for candidate in tried:
            if self._cache_dir_has_sample_files(candidate, patient_ids):
                if cache_dir and candidate != str(cache_dir):
                    log.info(f"Using fallback image cache directory: {candidate}")
                return candidate

        raise FileNotFoundError(
            "Unable to locate image cache with required patient files. "
            f"Tried: {', '.join(tried)}"
        )

    @staticmethod
    def _cache_dir_has_sample_files(cache_dir, patient_ids):
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            return False

        sample_ids = list(patient_ids)[:10]
        if not sample_ids:
            return True

        for patient_id in sample_ids:
            image_path = cache_dir / f"{patient_id}_image.nii.gz"
            mask_path = cache_dir / f"{patient_id}_mask_GTV.nii.gz"
            if not image_path.exists() or not mask_path.exists():
                return False

        return True

    def load_training_time_bines(self, patient_split, path, outcome, follow_up, time_bins):
        """Load time-bin boundaries from the reference training cohort."""
        candidate = self._reference_training_csv(path)
        data = pd.read_csv(candidate)
        data = data.rename(columns={"Death": "event1", "RT2Follow": "time1",\
            "LF": "event2", "RT2LF": "time2",\
            "RF": "event3", "RT2RF": "time3",\
            "DF": "event4", "RT2DF": "time4"})
        if patient_split:
            time_bins = make_time_bins(times=data[follow_up], num_bins=time_bins)
        else:
            time_bins = make_time_bins(times=data[follow_up], num_bins=time_bins, event = data[outcome])
        
        return time_bins

    def make_data(self, path):
        try:
            csv_path = self._resolve_data_csv(path)
            log.info(f"Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Rename columns
            df = df.rename(columns={
                "Death": "event1", 
                "RT2Follow": "time1",
                "LF": "event2", 
                "RT2LF": "time2",
                "RF": "event3", 
                "RT2RF": "time3",
                "DF": "event4", 
                "RT2DF": "time4"
            })
            
            # Drop any unnecessary columns
            cols_to_drop = []
            df = df.drop(cols_to_drop, axis=1)
            df = self._align_clinical_columns(df)
            
            return df
            
        except Exception as e:
            log.error(f'Error loading data from {path}: {str(e)}')
            raise  # Re-raise the exception to handle it at a higher level

    def _reference_training_csv(self, path):
        if isinstance(path, str) and path.endswith('.csv'):
            base_dir = os.path.dirname(path)
        else:
            base_dir = path

        candidates = []
        if base_dir:
            candidates.append(os.path.join(base_dir, self.train_filename))
        for fallback_dir in self._fallback_clinical_dirs():
            candidates.append(os.path.join(fallback_dir, self.train_filename))

        tried = list(self._unique_existing_path_candidates(candidates))
        for candidate in tried:
            if os.path.exists(candidate):
                if base_dir and candidate != os.path.join(base_dir, self.train_filename):
                    log.info(f"Using fallback training CSV for time bins: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Unable to locate {self.train_filename}. Tried: {', '.join(tried)}"
        )

    def _clinical_feature_columns(self, path):
        reference_df = pd.read_csv(self._reference_training_csv(path))
        reference_df = reference_df.rename(columns={
            "Death": "event1",
            "RT2Follow": "time1",
            "LF": "event2",
            "RT2LF": "time2",
            "RF": "event3",
            "RT2RF": "time3",
            "DF": "event4",
            "RT2DF": "time4",
        })
        return [col for col in reference_df.columns if col not in self.LABEL_COLUMNS]

    def _align_clinical_columns(self, df):
        for col in self.clinical_feature_columns:
            if col not in df.columns:
                df[col] = 0

        ordered_columns = []
        if 'ID' in df.columns:
            ordered_columns.append('ID')

        for col in self.clinical_feature_columns:
            ordered_columns.append(col)

        for col in self.LABEL_COLUMNS[1:]:
            if col in df.columns:
                ordered_columns.append(col)

        remaining_columns = [col for col in df.columns if col not in ordered_columns]
        return df[ordered_columns + remaining_columns]

    def __getitem__(self, idx: int):
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        
        clin_var_data = self.clinical_data[self.clinical_feature_columns]


        try:
            clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        except:
            log.info("Clinical features must be convertible to float32.")
        target = (self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx])
        available_label_columns = [col for col in self.LABEL_COLUMNS if col in self.clinical_data.columns]
        labels = self.clinical_data.iloc[idx][available_label_columns].to_dict()
        labels['time_bins1'] = self.time_bins1
        labels['time_bins2'] = self.time_bins2
        labels['time_bins3'] = self.time_bins3
        labels['time_bins4'] = self.time_bins4
        if self.domain_label is not None:
            labels['domain'] = torch.tensor(self.domain_label, dtype=torch.int64)
        if self.mini_batch_assignments is not None and self.mini_batch_assignments[idx] is not None and self.mini_batch_assignments[idx] != -1:
            labels['mini_batch_id'] = torch.tensor(int(self.mini_batch_assignments[idx]), dtype=torch.int64)
        
        
        sample = dict()
        
        id_ = self.cache_path[idx][0].parent.stem

        sample['id'] = id_
        img = [self.read_data(self.cache_path[idx][i]) for i in range(self.num_of_seqs)]
        img = np.stack(img, axis=-1)
        sample['input'] = img 
        
        mask = self.read_data(self.cache_path[idx][-1])
        mask = mask/255
        mask = np.expand_dims(mask, axis=3)
        sample['target_mask'] = mask

        
        if self.transforms:
            sample = self.transforms(sample)
        return (sample, clin_var), target, labels
    
    

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)
    
    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        if return_numpy:
            return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))
        return sitk.ReadImage(str(path_to_nifti))

    @staticmethod
    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        y = np.eye(num_classes+1, dtype='uint8')[y]
        return y[:,:,:,1:num_classes+1]
