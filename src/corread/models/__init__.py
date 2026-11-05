# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""The networks: CORnet-Z and CORnet-S, each as an image classifier and as two literate networks.

- ``pre`` and ``illit``: :class:`CORnetZ`, :class:`CORnetS`
- ``lit_bias``: :class:`CORnetZBiasedWords`, :class:`CORnetSBiasedWords`
- ``lit_no_bias``: :class:`CORnetZNonbiasedWords`, :class:`CORnetSNonbiasedWords`

Every network returns the activations of all its areas with the outputs; see :meth:`CORnet.forward`.
"""

from __future__ import annotations

from .base import BiasedWordClassifier, CORnet, ImageClassifier, UnbiasedWordClassifier
from .cornet_s import CORnetS, CORnetSBiasedWords, CORnetSNonbiasedWords
from .cornet_z import CORnetZ, CORnetZBiasedWords, CORnetZNonbiasedWords
from .layers import DENSE_UNITS, WORD_UNITS

NETWORKS: dict[str, dict[str, type[CORnet]]] = {
    'z': {'pre': CORnetZ, 'illit': CORnetZ, 'lit_bias': CORnetZBiasedWords, 'lit_no_bias': CORnetZNonbiasedWords},
    's': {'pre': CORnetS, 'illit': CORnetS, 'lit_bias': CORnetSBiasedWords, 'lit_no_bias': CORnetSNonbiasedWords},
}
"""The network class of each model choice (``'z'``, ``'s'``) and training mode."""


def network_class(model_choice: str, mode: str) -> type[CORnet]:
    """The class of the network that ``mode`` trains for CORnet-``model_choice``."""
    try:
        return NETWORKS[model_choice][mode]
    except KeyError:
        raise ValueError('No network for model_choice=' + repr(model_choice) + ', mode=' + repr(mode)) from None


def build_network(
    model_choice: str,
    mode: str,
    img_classes: int = 1000,
    wrd_classes: int = 1000,
    pretrained: ImageClassifier | None = None,
) -> CORnet:
    """A new network of a mode, optionally started from a pre-literate network (literate modes only)."""
    cls = network_class(model_choice, mode)
    if issubclass(cls, ImageClassifier):
        return cls(out_img=img_classes)
    return cls(pretrained, out_img=img_classes, out_wrd=wrd_classes)


__all__ = [
    'DENSE_UNITS',
    'NETWORKS',
    'WORD_UNITS',
    'BiasedWordClassifier',
    'CORnet',
    'CORnetS',
    'CORnetSBiasedWords',
    'CORnetSNonbiasedWords',
    'CORnetZ',
    'CORnetZBiasedWords',
    'CORnetZNonbiasedWords',
    'ImageClassifier',
    'UnbiasedWordClassifier',
    'build_network',
    'network_class',
]
