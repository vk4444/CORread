# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2018-2019 Jonas Kubilius and the DiCarlo Lab (CORnet)
# Copyright (C) 2021 T. Hannagan, A. Agrawal, L. Cohen, S. Dehaene (Origins-of-VWFA)
# Copyright (C) 2026 Vojtěch Kubala
#
# CORBlockZ, CORBlockS, the CORnet-S V1 block, Flatten and Identity are adapted from
# https://github.com/dicarlolab/CORnet (cornet_z.py, cornet_s.py); BiLinear from
# https://github.com/THANNAGA/Origins-of-VWFA (clean_cornets.py), via Dr. Olga Solaja's
# adaptation. Modified by Vojtěch Kubala, 2026: renamed to PEP 8 names, ReLUs not in place,
# and the readout made reusable across networks.
"""The building blocks of the CORnet networks and their readouts."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

DENSE_UNITS = 512
"""Units of the dense layer: IT's channels averaged over space, the input of every readout."""

WORD_UNITS = 49
"""Dense-layer units the word outputs of a biased network read: the last 49 (463..511)."""


class Flatten(nn.Module):
    """Flatten every dimension but the batch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Pass the input through unchanged; a named point to read a block's output from."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BiLinear(nn.Module):
    """The readout of a biased network.

    The image outputs read every dense-layer unit (``lin_img``); the word outputs read only the last
    ``in_wrd`` units (``lin_wrd``), the network's word channel. The outputs are the image outputs
    followed by the word outputs.
    """

    def __init__(self, in_img: int, out_img: int, in_wrd: int, out_wrd: int):
        super().__init__()
        self.lin_img = nn.Linear(in_img, out_img)
        self.lin_wrd = nn.Linear(in_wrd, out_wrd)
        self.in_wrd = in_wrd

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.lin_img(inp), self.lin_wrd(inp[:, -self.in_wrd :])), 1)


class CORBlockZ(nn.Module):
    """A CORnet-Z area: convolution, ReLU and max pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.output(self.pool(self.nonlin(self.conv(inp))))


class CORBlockS(nn.Module):
    """A recurrent CORnet-S area: a bottleneck residual block applied ``times`` times with shared weights.

    The first pass downsamples (stride 2) and adds a projected skip connection; the later passes keep
    the resolution and add the identity. Each pass has its own batch normalization.
    """

    scale = 4

    def __init__(self, in_channels: int, out_channels: int, times: int = 1):
        super().__init__()
        self.times = times
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels * self.scale, out_channels * self.scale, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.nonlin2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU()
        self.output = Identity()
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(inp)
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)
            x = self.nonlin1(getattr(self, f'norm1_{t}')(self.conv1(x)))
            x = self.nonlin2(getattr(self, f'norm2_{t}')(self.conv2(x)))
            x = getattr(self, f'norm3_{t}')(self.conv3(x))
            x += skip
            x = self.nonlin3(x)
            output = self.output(x)
        return output


def cornet_s_v1() -> nn.Sequential:
    """V1 of CORnet-S: two convolutions with a max pool between them."""
    return nn.Sequential(
        OrderedDict(
            [
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm1', nn.BatchNorm2d(64)),
                ('nonlin1', nn.ReLU()),
                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm2', nn.BatchNorm2d(64)),
                ('nonlin2', nn.ReLU()),
                ('output', Identity()),
            ]
        )
    )
