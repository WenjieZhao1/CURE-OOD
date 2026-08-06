"""
MLS (Maximum Logit Score) Postprocessor.

Unlike MSP which uses softmax probabilities, MLS directly uses
the maximum logit value as the OOD score.
"""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class MLSPostprocessor(BasePostprocessor):
    """
    Maximum Logit Score (MLS) postprocessor.

    Uses the maximum logit value (before softmax) as the confidence score.
    This is simpler than MSP and can be more effective in some cases.
    """
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        Compute MLS score.

        Args:
            net: The model
            data: Input data

        Returns:
            pred: Predicted class (argmax of logits)
            conf: Maximum logit value (OOD score)
        """
        output = net(data)
        # Use maximum logit directly (no softmax)
        conf, pred = torch.max(output, dim=1)
        return pred, conf
