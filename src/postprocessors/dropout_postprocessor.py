from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base_postprocessor import BasePostprocessor


class DropoutPostProcessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args
        self.dropout_times = self.args.dropout_times
        self.dropout_p = self.args.dropout_p

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        MC Dropout: Apply dropout to features and average predictions.

        Unlike OpenMIBOOD which uses DropoutNet wrapper, we directly apply
        dropout in the postprocessor by:
        1. Extract features from backbone
        2. Apply dropout to features (with training=True)
        3. Compute logits from dropped features
        4. Repeat multiple times and average logits
        """
        logits_list = []
        for _ in range(self.dropout_times):
            # Get features from the model
            features = net._get_features(data)
            # Apply dropout (training=True enables dropout during inference)
            features_dropped = F.dropout(features, p=self.dropout_p, training=True)
            # Compute logits from dropped features
            logits = net._compute_head_logits(features_dropped)
            logits_list.append(logits)

        # Average logits across all dropout samples
        logits_mean = torch.stack(logits_list).mean(dim=0)

        # Compute confidence score
        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)

        return pred, conf
