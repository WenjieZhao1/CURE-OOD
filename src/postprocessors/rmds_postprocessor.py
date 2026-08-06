from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _extract_data_and_labels(batch, device):
    if isinstance(batch, dict):
        data = _move_to_device(batch['data'], device)
        label = batch.get('label')
        label = label.to(device) if isinstance(label, torch.Tensor) else label
        data = data.float() if isinstance(data, torch.Tensor) else data
        return data, label
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        data = batch[0]
        if isinstance(data, (list, tuple)) and len(data) == 2:
            sample, clin_var = data
            sample = _move_to_device(sample, device)
            clin_var = _move_to_device(clin_var, device)
            data = (sample, clin_var)
        else:
            data = _move_to_device(data, device)
        return data, None
    return batch, None


class RMDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = None
        self.active_classes = None
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            all_feats = []
            all_labels = []
            all_preds = []
            device = next(net.parameters(), torch.tensor(0.)).device
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = _extract_data_and_labels(batch, device)
                    logits, features = net(data, return_feature=True)
                    if self.num_classes is None:
                        self.num_classes = logits.shape[1]
                    if labels is None:
                        labels = logits.argmax(1)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels).cpu())
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            if self.num_classes is None:
                self.num_classes = all_preds.unique().numel()
            # compute class-conditional statistics
            self.active_classes = torch.unique(all_labels).tolist()
            self.class_mean = []
            centered_data = []
            for c in self.active_classes:
                class_samples = all_feats[all_labels.eq(c)].data
                if class_samples.numel() == 0:
                    continue
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[-1].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()

            self.whole_mean = all_feats.mean(0)
            centered_data = all_feats - self.whole_mean.view(1, -1)
            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            self.whole_precision = torch.from_numpy(
                group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)

        tensor1 = features.cpu() - self.whole_mean.view(1, -1)
        background_scores = -torch.matmul(
            torch.matmul(tensor1, self.whole_precision), tensor1.t()).diag()

        class_scores = torch.full((logits.shape[0], len(self.active_classes)), float("-inf"))
        for idx, _ in enumerate(self.active_classes):
            tensor = features.cpu() - self.class_mean[idx].view(1, -1)
            class_scores[:, idx] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()
            class_scores[:, idx] = class_scores[:, idx] - background_scores

        conf = torch.max(class_scores, dim=1)[0]
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1e6, neginf=-1e6)
        return pred, conf
