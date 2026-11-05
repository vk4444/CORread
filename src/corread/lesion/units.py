# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""The dense-layer units of a literate CORnet-Z: their word selectivity, and lesions of them.

The dense layer is IT averaged over space: 512 units, numbered like IT's channels. In the biased
network the last 49 (463..511) are the only input of the word outputs, its *word channel*.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from ..models import DENSE_UNITS


def word_channel(net: nn.Module) -> tuple[int, int] | None:
    """``(first, last)`` unit the biased network's word outputs read; ``None`` for the unbiased network."""
    if not hasattr(net, 'bilinear'):
        return None
    n = net.IT.conv.out_channels
    return n - net.bilinear.lin_wrd.in_features, n - 1


def _readouts(net: nn.Module, units: list[int]) -> list[tuple[nn.Linear, list[int]]]:
    """``(readout layer, its input columns of the given units)`` of each readout layer the units feed."""
    channel = word_channel(net)
    if channel is None:
        return [(net.linear, list(units))]
    readouts = [(net.bilinear.lin_img, list(units))]
    word_columns = [u - channel[0] for u in units if u >= channel[0]]
    if word_columns:
        readouts.append((net.bilinear.lin_wrd, word_columns))
    return readouts


def _uniform(shape: tuple[int, ...], bound: float, generator: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    """U(-bound, bound) values drawn on the CPU from ``generator``, as the dtype and on the device of ``like``."""
    return torch.empty(shape).uniform_(-bound, bound, generator=generator).to(like.device, like.dtype)


def lesion(
    net: nn.Module, units: list[int], reset_outgoing: bool, generator: torch.Generator
) -> tuple[float | None, float | None]:
    """Re-initialise the incoming weights of dense-layer units, as PyTorch initialises an untrained layer.

    Each unit ``u`` gets a new IT conv filter ``u`` and bias, drawn from U(-1/sqrt(fan_in), 1/sqrt(fan_in));
    with ``reset_outgoing`` also new weights in the readout, U(-1/sqrt(in_features), 1/sqrt(in_features)).
    The output biases stay. Returns the mean norm of the units' filters before and after (``None``
    without units).
    """
    if not units:
        return None, None
    conv = net.IT.conv
    with torch.no_grad():
        index = torch.tensor(units, device=conv.weight.device)
        before = conv.weight[index].flatten(1).norm(dim=1).mean().item()
        bound = 1 / math.sqrt(conv.weight[0].numel())
        conv.weight[index] = _uniform((len(units),) + tuple(conv.weight.shape[1:]), bound, generator, conv.weight)
        conv.bias[index] = _uniform((len(units),), bound, generator, conv.bias)
        after = conv.weight[index].flatten(1).norm(dim=1).mean().item()
        if reset_outgoing:
            for linear, columns in _readouts(net, units):
                bound = 1 / math.sqrt(linear.in_features)
                cols = torch.tensor(columns, device=linear.weight.device)
                linear.weight[:, cols] = _uniform((linear.out_features, len(columns)), bound, generator, linear.weight)
    return before, after


def _whole(value, minimum: int) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= minimum


def check_units(units: Iterable[int]) -> list[int]:
    """The units to lesion as a sorted list of distinct dense-layer indices."""
    if isinstance(units, (str, bytes)) or not hasattr(units, '__iter__'):
        raise ValueError('units must be a list of unit indices (0..' + str(DENSE_UNITS - 1) + '), got: ' + repr(units))
    checked = []
    for u in units:
        if not _whole(u, 0) or u >= DENSE_UNITS:
            raise ValueError('units must be unit indices from 0 to ' + str(DENSE_UNITS - 1) + ', got: ' + repr(u))
        checked.append(int(u))
    repeated = sorted(u for u in set(checked) if checked.count(u) > 1)
    if repeated:
        raise ValueError('units lists these units more than once: ' + repr(repeated))
    return sorted(checked)


def selectivity(mean_words: np.ndarray, mean_images: np.ndarray, sd_images: np.ndarray) -> np.ndarray:
    """How many SDs of its responses to images each unit's mean response to words lies above its mean to images.

    An SD of 0 gives ±inf, or nan when the two means are equal.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return (mean_words - mean_images) / sd_images


def selective(values: np.ndarray, threshold: float) -> np.ndarray:
    """Which units are word-selective (nan is not)."""
    with np.errstate(invalid='ignore'):
        return values >= threshold


class UnitSums:
    """Running sums of the dense layer's responses per unit, apart for images and words.

    Called with each batch's dense-layer activations ``h`` and labels; labels below ``img_classes`` are images.
    """

    def __init__(self, img_classes: int):
        self.img_classes = img_classes
        self.n: dict[str, int] = {}
        self.sums: dict[str, torch.Tensor] = {}
        self.squares: dict[str, torch.Tensor] = {}

    def __call__(self, h: torch.Tensor, labels: torch.Tensor) -> None:
        h = h.detach().double()
        words = labels >= self.img_classes
        for kind, mask in (('images', ~words), ('words', words)):
            x = h[mask]
            if len(x):
                self.n[kind] = self.n.get(kind, 0) + len(x)
                self.sums[kind] = self.sums.get(kind, 0) + x.sum(0)
                self.squares[kind] = self.squares.get(kind, 0) + (x * x).sum(0)

    def count(self, kind: str) -> int:
        return self.n.get(kind, 0)

    def mean(self, kind: str) -> np.ndarray:
        return (self.sums[kind] / self.n[kind]).cpu().numpy()

    def sd(self, kind: str) -> np.ndarray:
        """The SD of the responses (ddof=1)."""
        n = self.n[kind]
        var = (self.squares[kind] - self.sums[kind] ** 2 / n) / (n - 1)
        return var.clamp(min=0).sqrt().cpu().numpy()
