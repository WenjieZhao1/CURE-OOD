from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


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


class DICEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DICEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.p = self.args.p
        self.mean_act = None
        self.masked_w = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            device = next(net.parameters(), torch.tensor(0.)).device
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = _extract_data(batch, device)
                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            activation_log = np.concatenate(activation_log, axis=0)
            self.mean_act = activation_log.mean(0)
            self.setup_flag = True
        else:
            pass

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.p)
        mask = torch.as_tensor((contrib > self.thresh), device=w.device, dtype=w.dtype)
        self.masked_w = w * mask

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        fc_weight, fc_bias = net.get_fc()
        if self.masked_w is None:
            if not isinstance(fc_weight, torch.Tensor):
                fc_weight = torch.from_numpy(fc_weight)
            device = next(net.parameters(), torch.tensor(0.)).device
            self.calculate_mask(fc_weight.to(device))
        _, feature = net(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        if not isinstance(fc_bias, torch.Tensor):
            fc_bias = torch.from_numpy(fc_bias)
        output = vote.sum(2) + fc_bias.to(vote.device)
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]
        # With this, it is ensured that the mask is recalculated with the final selected hyperparameter
        self.masked_w = None

    def get_hyperparam(self):
        return self.p
