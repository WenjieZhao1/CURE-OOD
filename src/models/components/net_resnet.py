import torch
from torch import nn
from torchmtlr import MTLR

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_MTLR(nn.Module):

    def __init__(self,
                 hparams,
                 block_inplanes=[64, 128, 256, 512],
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()
        
        model_depth = hparams['model_depth']
        if model_depth == 10:
            block = BasicBlock
            layers = [1, 1, 1, 1]
        elif model_depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif model_depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif model_depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif model_depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif model_depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        elif model_depth == 200:
            block = Bottleneck
            layers = [3, 24, 36, 3]

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.n_clin_var = int(hparams.get('n_clin_var', 0) or 0)

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if hparams['n_dense'] <=0:
            
            self.fc_layers1 = nn.Dropout(hparams['dropout']) 
            
            self.mtlr1 = MTLR(block_inplanes[3] * block.expansion + self.n_clin_var, hparams['time_bins'])
            self.mtlr2 = MTLR(block_inplanes[3] * block.expansion + self.n_clin_var, hparams['time_bins'])
            self.mtlr3 = MTLR(block_inplanes[3] * block.expansion + self.n_clin_var, hparams['time_bins'])
            self.mtlr4 = MTLR(block_inplanes[3] * block.expansion + self.n_clin_var, hparams['time_bins'])

        elif hparams['n_dense'] ==1:
            self.fc_layers1 = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion + self.n_clin_var, 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            
        elif hparams['n_dense'] > 1:    
            self.fc_layers1 = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion + self.n_clin_var , 128*hparams['dense_factor']), 
                          nn.BatchNorm1d(128*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']),
                          nn.Linear(128*hparams['dense_factor'] , 64*hparams['dense_factor']), 
                          nn.BatchNorm1d(64*hparams['dense_factor']),
                          nn.ReLU(inplace=True), 
                          nn.Dropout(hparams['dropout']))
            
            self.mtlr1 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr2 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr3 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])
            self.mtlr4 = MTLR(64*hparams['dense_factor'], hparams['time_bins'])


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _prepare_image_input(self, sample_img):
        return torch.cat(
            (sample_img['target_mask'][:, 0:1, :], sample_img['input'][:, 0:1, :]),
            dim=1,
        )

    def _extract_backbone_features(self, sample_img):
        x = self._prepare_image_input(sample_img)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def _split_sample(self, sample):
        if isinstance(sample, (tuple, list)):
            if len(sample) >= 2:
                return sample[0], sample[1]
            if len(sample) == 1:
                return sample[0], None
        return sample, None

    def get_features(self, sample):
        sample_img, clin_var = self._split_sample(sample)
        x = self._extract_backbone_features(sample_img)
        if self.n_clin_var <= 0:
            return x
        if clin_var is None:
            raise ValueError("Clinical variables are required when n_clin_var > 0.")
        return torch.cat((x, clin_var), dim=1)

    def get_mtlr_features(self, sample):
        x = self.get_features(sample)
        return self.fc_layers1(x)

    def forward(self, x):
        x = self.get_mtlr_features(x)

        risk_out1, _ = self.mtlr1(x)
        risk_out2, _ = self.mtlr2(x)
        risk_out3, _ = self.mtlr3(x)
        risk_out4, _ = self.mtlr4(x)
        
        return risk_out1,  risk_out2, risk_out3, risk_out4
