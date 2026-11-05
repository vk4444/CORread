# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2019 Jonas Kubilius and the DiCarlo Lab (CORnet)
# Copyright (C) 2026 Vojtěch Kubala
#
# The CORnet-S architecture follows https://github.com/dicarlolab/CORnet (cornet_s.py).
# Modified by Vojtěch Kubala, 2026: built from the shared networks of corread.models.base,
# with the literate readouts of the CORnet-Z networks.
"""CORnet-S (Kubilius et al., 2019): recurrent, residual areas with batch normalization."""

from __future__ import annotations

from torch import nn

from .base import BiasedWordClassifier, ImageClassifier, UnbiasedWordClassifier
from .layers import CORBlockS, cornet_s_v1


def cornet_s_areas() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """New V1, V2, V4 and IT areas of CORnet-S (V2 and IT unrolled twice, V4 four times)."""
    return cornet_s_v1(), CORBlockS(64, 128, times=2), CORBlockS(128, 256, times=4), CORBlockS(256, 512, times=2)


class CORnetS(ImageClassifier):
    """CORnet-S reading out to the image classes only: the pre-literate and illiterate network."""

    backbone = staticmethod(cornet_s_areas)


class CORnetSBiasedWords(BiasedWordClassifier):
    """Literate CORnet-S whose word outputs read only the last 49 dense-layer units."""

    backbone = staticmethod(cornet_s_areas)


class CORnetSNonbiasedWords(UnbiasedWordClassifier):
    """Literate CORnet-S whose word outputs read every dense-layer unit."""

    backbone = staticmethod(cornet_s_areas)
