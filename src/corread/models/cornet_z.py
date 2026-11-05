# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2018 Jonas Kubilius and the DiCarlo Lab (CORnet)
# Copyright (C) 2026 Vojtěch Kubala
#
# The CORnet-Z architecture follows https://github.com/dicarlolab/CORnet (cornet_z.py).
# Modified by Vojtěch Kubala, 2026: built from the shared networks of corread.models.base.
"""CORnet-Z (Kubilius et al., 2018): four feedforward areas of one convolution each."""

from __future__ import annotations

from torch import nn

from .base import BiasedWordClassifier, ImageClassifier, UnbiasedWordClassifier
from .layers import CORBlockZ


def cornet_z_areas() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """New V1, V2, V4 and IT areas of CORnet-Z."""
    return CORBlockZ(3, 64, kernel_size=7, stride=2), CORBlockZ(64, 128), CORBlockZ(128, 256), CORBlockZ(256, 512)


class CORnetZ(ImageClassifier):
    """CORnet-Z reading out to the image classes only: the pre-literate and illiterate network."""

    backbone = staticmethod(cornet_z_areas)


class CORnetZBiasedWords(BiasedWordClassifier):
    """Literate CORnet-Z whose word outputs read only the last 49 dense-layer units."""

    backbone = staticmethod(cornet_z_areas)


class CORnetZNonbiasedWords(UnbiasedWordClassifier):
    """Literate CORnet-Z whose word outputs read every dense-layer unit."""

    backbone = staticmethod(cornet_z_areas)
