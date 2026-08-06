"""
SCALE (Scaling with pruning) wrapper for adapter.

This wrapper provides the forward_threshold method required by SCALE postprocessor.
"""
import numpy as np
import torch
import torch.nn as nn


def scale(x, percentile=65):
    """
    SCALE: Scaling features based on pruning statistics.

    Unlike ASH which modifies the pruned features, SCALE:
    1. Clones the original input
    2. Prunes a copy to calculate scaling statistics
    3. Applies the scaling to the ORIGINAL input

    Args:
        x: Input features with shape (b, c, h, w)
        percentile: Percentage of activations to prune (0-100)

    Returns:
        Scaled features (original features scaled by pruning statistics)
    """
    input = x.clone()  # Keep original input
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # Calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    # Calculate number of elements to keep
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))

    # Flatten and get top-k values (this modifies x)
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # Calculate new sum after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # Calculate scaling factor
    scale_factor = s1 / s2

    # Apply scaling to ORIGINAL input (not the pruned version)
    return input * torch.exp(scale_factor[:, None, None, None])


class ScaleWrapper(nn.Module):
    """
    Wrapper that adds SCALE support to the adapter.

    Similar to OpenMIBOOD's ScaleNet, this wrapper provides
    forward_threshold() method for SCALE postprocessor.
    """
    def __init__(self, adapter):
        super(ScaleWrapper, self).__init__()
        self.adapter = adapter

    def forward(self, x, return_feature=False, return_feature_list=False, return_per_task=False):
        """Standard forward pass (delegates to adapter)."""
        return self.adapter(x, return_feature, return_feature_list, return_per_task)

    def forward_threshold(self, data, percentile, return_per_task=False):
        """
        Forward pass with SCALE.

        Args:
            data: Input data
            percentile: Percentage of activations to prune for scaling (0-100)
            return_per_task: If True, return per-task logits instead of concatenated

        Returns:
            Logits after SCALE processing
        """
        # Get features from the adapter
        features = self.adapter._get_features(data)

        # Ensure features are 4D for scale
        # If features are 2D (batch, features), reshape to (batch, features, 1, 1)
        if features.dim() == 2:
            features = features.view(features.size(0), -1, 1, 1)

        # Apply SCALE
        features_scaled = scale(features, percentile)

        # Flatten back to 2D for FC layer
        features_scaled = features_scaled.view(features_scaled.size(0), -1)

        # Compute logits from scaled features
        logits = self.adapter._compute_head_logits(features_scaled, return_per_task=return_per_task)

        return logits

    def _get_features(self, data):
        """Delegate to adapter."""
        return self.adapter._get_features(data)

    def _compute_head_logits(self, features, return_per_task=False):
        """Delegate to adapter."""
        return self.adapter._compute_head_logits(features, return_per_task=return_per_task)

    def get_fc(self):
        """Delegate to adapter."""
        return self.adapter.get_fc()

    def __getattr__(self, name):
        """Forward any other attributes to the adapter."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.adapter, name)
