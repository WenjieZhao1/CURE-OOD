from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Task names for better readability
TASK_NAMES = ["OS", "LFFS", "RFFS", "DFFS"]
def _gather_mtlr_heads(module: nn.Module) -> List[nn.Module]:
    heads = []
    for name, submodule in module.named_modules():
        if hasattr(submodule, "mtlr_weight"):
            heads.append((name, submodule))
    heads.sort(key=lambda item: item[0])
    unique = []
    seen = set()
    for name, head in heads:
        if id(head) in seen:
            continue
        seen.add(id(head))
        unique.append(head)
    return unique


def _ensure_tensor_dict(data):
    return data


class SurvivalModelAdapter(nn.Module):
    """Adapter exposing a classification-like interface expected by post-hoc OOD detectors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.backbone = getattr(model, "model", model)
        self.deephit_head = getattr(model, "deephit", None)
        # A DeepHit model is identified by its dedicated `deephit` head. Its
        # backbone (UNETR_Img) still carries unused mtlr1..mtlr4 submodules, so
        # we must NOT fall back to those MTLR heads when a DeepHit head exists.
        if self.deephit_head is not None:
            self.is_deephit = True
            self.heads = []
        else:
            self.is_deephit = False
            self.heads = _gather_mtlr_heads(self.backbone)

        self.time_bins = []
        if self.is_deephit:
            self.time_bins.append(self.deephit_head.out_features)
        else:
            if not self.heads:
                raise ValueError("Unable to locate MTLR or DeepHit heads on the provided model.")
            for head in self.heads:
                bins = getattr(head, "num_time_bins", None)
                if bins is None:
                    weight = getattr(head, "mtlr_weight", None)
                    if weight is None:
                        raise ValueError("MTLR head does not expose num_time_bins or mtlr_weight.")
                    bins = weight.shape[1] + 1
                self.time_bins.append(bins)
        self.num_classes = sum(self.time_bins)

    def _compute_head_logits(self, features: torch.Tensor, return_per_task: bool = False):
        """
        Compute logits from all MTLR heads.

        Args:
            features: Input features (batch, feature_dim)
            return_per_task: If True, return list of per-task logits.
                           If False, return concatenated logits.

        Returns:
            If return_per_task=False: Concatenated logits (batch, total_time_bins)
            If return_per_task=True: List of [logits1, logits2, logits3, logits4]
        """
        if self.is_deephit:
            logits = [self.deephit_head(features)]
            return logits if return_per_task else logits[0]

        logits = []
        for head in self.heads:
            out = head(features)
            if isinstance(out, (tuple, list)):
                head_logits = out[-1]
            else:
                head_logits = out
            logits.append(head_logits)

        if return_per_task:
            return logits  # Return list of task logits
        else:
            return torch.cat(logits, dim=1)  # Return concatenated logits

    def _get_features(self, data):
        if isinstance(data, torch.Tensor):
            return data
        sample, clin_var = data
        if hasattr(self.backbone, "get_mtlr_features"):
            features = self.backbone.get_mtlr_features((sample, clin_var))
        elif hasattr(self.backbone, "get_features"):
            features = self.backbone.get_features((sample, clin_var))
        else:
            raise AttributeError("Backbone must expose get_mtlr_features or get_features.")
        return features.float()

    def forward(self,
                data,
                return_feature: bool = False,
                return_feature_list: bool = False,
                return_per_task: bool = False):
        """
        Forward pass through the adapter.

        Args:
            data: Input data (sample, clin_var)
            return_feature: Return features along with logits
            return_feature_list: Return features as a list
            return_per_task: Return per-task logits instead of concatenated

        Returns:
            If return_per_task=False: logits (batch, 36) or (logits, features)
            If return_per_task=True: [logits1, logits2, logits3, logits4] or with features
        """
        features = self._get_features(_ensure_tensor_dict(data))
        logits = self._compute_head_logits(features, return_per_task=return_per_task)

        if return_feature and return_feature_list:
            return logits, [features]
        if return_feature:
            return logits, features
        if return_feature_list:
            return logits, [features]
        return logits

    def forward_threshold(self, data, percentile: float, return_per_task: bool = False):
        """Forward with threshold (used by ASH, ReAct, etc.)."""
        features = self._get_features(_ensure_tensor_dict(data))
        return self._compute_head_logits(features, return_per_task=return_per_task)

    def get_fc(self):
        """
        Extract the weight and bias from all MTLR heads and concatenate them.
        Returns (w, b) where logits = features @ w.T + b
        """
        w_list = []
        b_list = []

        if self.is_deephit:
            return self.deephit_head.weight.detach().cpu(), self.deephit_head.bias.detach().cpu()

        for head in self.heads:
            # Extract mtlr_weight from the head
            if hasattr(head, 'mtlr_weight'):
                weight = head.mtlr_weight  # Shape: (input_features, num_time_bins - 1)
                # Transpose to (num_time_bins - 1, input_features) for standard FC format
                w_list.append(weight.t().detach().cpu().numpy())

                # Extract bias if exists
                if hasattr(head, 'mtlr_bias') and head.mtlr_bias is not None:
                    bias = head.mtlr_bias.detach().cpu().numpy()
                else:
                    # If no bias, create zeros
                    bias = torch.zeros(weight.shape[1]).numpy()
                b_list.append(bias)
            else:
                raise AttributeError(f"MTLR head does not have mtlr_weight attribute")

        # Concatenate all weights and biases
        w = torch.from_numpy(np.vstack(w_list))  # Shape: (total_classes, input_features)
        b = torch.from_numpy(np.concatenate(b_list))  # Shape: (total_classes,)

        return w, b

    def get_fc_layer(self) -> nn.Linear:
        raise NotImplementedError("FC layer reconstruction is not implemented for the survival adapter.")

    def intermediate_forward(self, data, layer_index: int):
        features = self._get_features(_ensure_tensor_dict(data))
        return features.unsqueeze(-1).unsqueeze(-1)

    def avgpool(self, features):
        if features.dim() < 4:
            return features
        return features.mean(dim=list(range(2, features.dim())), keepdim=True)

    def layer4(self, features):
        return features
