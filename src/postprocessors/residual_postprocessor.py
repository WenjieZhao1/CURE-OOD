from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
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


def _extract_data(batch, device):
    if isinstance(batch, dict):
        data = _move_to_device(batch['data'], device)
        return data.float() if isinstance(data, torch.Tensor) else data
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        data = batch[0]
        if isinstance(data, (list, tuple)) and len(data) == 2:
            sample, clin_var = data
            sample = _move_to_device(sample, device)
            clin_var = _move_to_device(clin_var, device)
            return (sample, clin_var)
        return _move_to_device(data, device)
    return batch


class ResidualPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
    
    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]
        self._update_internals()

    def get_hyperparam(self):
        return self.dim

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()
        device = next(net.parameters(), torch.tensor(0.)).device

        with torch.no_grad():
            self.w, self.b = net.get_fc()
            if isinstance(self.w, torch.Tensor):
                self.w = self.w.detach().cpu().numpy()
            if isinstance(self.b, torch.Tensor):
                self.b = self.b.detach().cpu().numpy()
            feature_id_train = []
            for batch in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = _extract_data(batch, device)
                _, feature = net(data, return_feature=True)
                feature_id_train.append(feature.cpu().numpy())
            feature_id_train = np.concatenate(feature_id_train, axis=0)

            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = _extract_data(batch, device)
                _, feature = net(data, return_feature=True)
                feature_id_val.append(feature.cpu().numpy())
            feature_id_val = np.concatenate(feature_id_val, axis=0)

        self.u = -np.matmul(pinv(self.w), self.b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - self.u)
        self.eig_vals, self.eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray(
            (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.dim:]]).T)

        self.score_id = -norm(np.matmul(feature_id_val - self.u, self.NS),
                              axis=-1)
        
    def _update_internals(self):
        self.NS = np.ascontiguousarray(
            (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.dim:]]).T)


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net(data, return_feature=True)
        feature_ood = feature_ood.cpu().numpy()
        logit_ood = torch.as_tensor(feature_ood @ self.w.T + self.b)
        _, pred = torch.max(logit_ood, dim=1)
        score_ood = -norm(np.matmul(feature_ood - self.u, self.NS),
                          axis=-1)
        return pred, torch.from_numpy(score_ood)
