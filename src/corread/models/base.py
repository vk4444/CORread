# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2021 T. Hannagan, A. Agrawal, L. Cohen, S. Dehaene (Origins-of-VWFA)
# Copyright (C) 2026 Vojtěch Kubala
#
# The literate and illiterate networks follow https://github.com/THANNAGA/Origins-of-VWFA
# (clean_cornets.py), via Dr. Olga Solaja's adaptation. Modified by Vojtěch Kubala, 2026: one
# shared forward pass and backbone for CORnet-Z and CORnet-S, and configurable output sizes.
"""The three kinds of network every training mode uses, independent of the CORnet variant.

Each network is a ventral stream (areas ``V1``, ``V2``, ``V4``, ``IT``), a global average pool that
turns IT into the 512-unit dense layer, and a readout from the dense layer to the outputs. The
variants (CORnet-Z, CORnet-S) only differ in the areas; see :mod:`corread.models.cornet_z` and
:mod:`corread.models.cornet_s`.

The networks use PyTorch's default initialization, not upstream CORnet's Xavier initialization.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .layers import DENSE_UNITS, WORD_UNITS, BiLinear, Flatten

Activations = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CORnet(nn.Module):
    """Base class: the ventral stream, the dense layer, and a readout defined by subclasses.

    Subclasses set :meth:`backbone`, which builds new areas, and implement :meth:`readout`.
    """

    @staticmethod
    def backbone() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """New, untrained V1, V2, V4 and IT areas."""
        raise NotImplementedError

    def _init_stream(self, pretrained: CORnet | None) -> None:
        # the areas of a pretrained network are shared, not copied: the new network trains them further
        if pretrained is None:
            self.V1, self.V2, self.V4, self.IT = self.backbone()
        else:
            self.V1, self.V2, self.V4, self.IT = pretrained.V1, pretrained.V2, pretrained.V4, pretrained.IT
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """The outputs (logits) from the dense-layer activations ``h``."""
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> Activations:
        """Run a batch of images through the network.

        Returns
        -------
        tuple
            The flattened activations of V1, V2, V4 and IT, the dense layer ``h`` (batch × 512) and the
            outputs (batch × number of classes).
        """
        v1 = self.V1(image)
        v2 = self.V2(v1)
        v4 = self.V4(v2)
        it = self.IT(v4)
        h = self.flatten(self.avgpool(it))
        return self.flatten(v1), self.flatten(v2), self.flatten(v4), self.flatten(it), h, self.readout(h)


class ImageClassifier(CORnet):
    """The pre-literate (and illiterate) network: every dense-layer unit reads out to the image classes.

    Parameters
    ----------
    out_img : int
        Number of image classes.
    """

    def __init__(self, out_img: int = 1000):
        super().__init__()
        self._init_stream(None)
        self.decoder = nn.Sequential(OrderedDict([('linear', nn.Linear(DENSE_UNITS, out_img))]))

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)


class BiasedWordClassifier(CORnet):
    """The biased literate network: the word outputs read only the last 49 dense-layer units.

    The image outputs read every unit, as in the pre-literate network; the word outputs read only its
    word channel, units 463..511 (:class:`~corread.models.layers.BiLinear`).

    Parameters
    ----------
    pretrained : ImageClassifier, optional
        The pre-literate network to start from: its areas are taken over and its image readout copied.
    out_img, out_wrd : int
        Number of image and word classes; the outputs are the image classes, then the words.
    """

    def __init__(self, pretrained: ImageClassifier | None = None, out_img: int = 1000, out_wrd: int = 1000):
        super().__init__()
        self._init_stream(pretrained)
        self.bilinear = BiLinear(in_img=DENSE_UNITS, out_img=out_img, in_wrd=WORD_UNITS, out_wrd=out_wrd)
        if pretrained is not None:
            with torch.no_grad():
                self.bilinear.lin_img.weight.copy_(pretrained.decoder.linear.weight)
                self.bilinear.lin_img.bias.copy_(pretrained.decoder.linear.bias)

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.bilinear(h)


class UnbiasedWordClassifier(CORnet):
    """The unbiased literate network: every dense-layer unit reads out to every output.

    Parameters
    ----------
    pretrained : ImageClassifier, optional
        The pre-literate network to start from: its areas are taken over and its readout copied into
        the image outputs.
    n_hid : int
        Units of the dense layer.
    out_img, out_wrd : int
        Number of image and word classes; the outputs are the image classes, then the words.
    """

    def __init__(
        self,
        pretrained: ImageClassifier | None = None,
        n_hid: int = DENSE_UNITS,
        out_img: int = 1000,
        out_wrd: int = 1000,
    ):
        super().__init__()
        self._init_stream(pretrained)
        self.linear = nn.Linear(n_hid, out_img + out_wrd)
        if pretrained is not None:
            with torch.no_grad():
                self.linear.weight[:out_img, :] = pretrained.decoder.linear.weight
                self.linear.bias[:out_img] = pretrained.decoder.linear.bias

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
