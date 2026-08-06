from typing import Any

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader


def _is_main_process() -> bool:
    """Heuristic replacement for OpenOOD's distributed check."""
    return True


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        iterator = tqdm(data_loader,
                        disable=not progress or not _is_main_process())
        param = next(net.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        for batch in iterator:
            data = batch['data']
            label = batch['label']

            data = _move_to_device(data, device)
            label = _move_to_device(label, device)

            pred, conf = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
