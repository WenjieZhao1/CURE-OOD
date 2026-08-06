"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any
import warnings

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .normalization import normalization_dict


class ODINPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.temperature = self.args.temperature
        self.noise = self.args.noise
        self.needs_grad = True
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
            warnings.warn('No normalization found for dataset. Using default values.', RuntimeWarning)
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def postprocess(self, net: nn.Module, data: Any):
        # Handle tuple input (sample_dict, clinical_vars)
        if isinstance(data, tuple):
            sample, clinical_vars = data
            # Extract image tensor from sample dict
            if isinstance(sample, dict):
                image = sample['input']
                image.requires_grad = True
                output = net((sample, clinical_vars))
            else:
                # sample is a tensor
                sample.requires_grad = True
                image = sample
                output = net((sample, clinical_vars))
        else:
            # Single tensor input
            data.requires_grad = True
            image = data
            output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(image.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        # 3D Data with one channel
        if gradient.shape[1] == 1 and len(gradient.shape) == 5:
            gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        else:
            gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
            gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
            gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(image.detach(), gradient, alpha=-self.noise)

        # Call net with perturbed input
        if isinstance(data, tuple):
            sample, clinical_vars = data
            if isinstance(sample, dict):
                # Create new sample dict with perturbed image
                perturbed_sample = sample.copy()
                perturbed_sample['input'] = tempInputs
                output = net((perturbed_sample, clinical_vars))
            else:
                # sample is a tensor
                output = net((tempInputs, clinical_vars))
        else:
            output = net(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]
