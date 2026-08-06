from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
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


class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False
        self.feature_id_train = []
        self.logit_id_train = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            device = next(net.parameters(), torch.tensor(0.)).device

            with torch.no_grad():
                self.w, self.b = net.get_fc()
                if isinstance(self.w, torch.Tensor):
                    self.w = self.w.detach().cpu().numpy()
                if isinstance(self.b, torch.Tensor):
                    self.b = self.b.detach().cpu().numpy()
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    # Handle both standard format (batch['data']) and survival format ((sample, clin_var), _, _)
                    if isinstance(batch, dict):
                        data = _move_to_device(batch['data'], device)
                        if isinstance(data, torch.Tensor):
                            data = data.float()
                    else:
                        # Survival analysis format: ((sample, clin_var), event, time)
                        (sample, clin_var), _, _ = batch
                        sample = _move_to_device(sample, device)
                        clin_var = _move_to_device(clin_var, device)
                        data = (sample, clin_var)

                    _, feature = net(data, return_feature=True)
                    self.feature_id_train.append(feature.cpu().numpy())
                self.feature_id_train = np.concatenate(self.feature_id_train, axis=0)
                self.logit_id_train = self.feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)

            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(self.feature_id_train - self.u)

            self.eig_vals, self.eigen_vectors = np.linalg.eig(ec.covariance_)
            self.calculate_params()


            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        logit_ood = torch.tensor(feature_ood @ self.w.T + self.b) # torch.tensor is necessary because of the metatensor with the OASIS3 dataset
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        score_ood = np.nan_to_num(score_ood, nan=0.0, posinf=1e6, neginf=-1e6)
        return pred, torch.from_numpy(score_ood)

    def calculate_params(self):
        feature_dim = int(self.feature_id_train.shape[1])
        effective_dim = min(max(int(self.dim), 1), max(feature_dim - 1, 1))
        self.NS = np.ascontiguousarray(
            (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[effective_dim:]]).T
        )
        vlogit_id_train = norm(np.matmul(self.feature_id_train - self.u, self.NS), axis=-1)
        denom = float(vlogit_id_train.mean())
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            denom = 1e-12
        numer = float(self.logit_id_train.max(axis=-1).mean())
        self.alpha = numer / denom
        if not np.isfinite(self.alpha):
            self.alpha = 1.0

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]
        self.calculate_params()

    def get_hyperparam(self):
        return self.dim
