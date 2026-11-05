# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""CORread: train CORnets to read, and study the word-selective units that emerge.

A convolutional network of the ventral visual stream (CORnet-Z or CORnet-S) is first trained to
recognize objects, then taught to read, after Hannagan et al. (2021). CORread makes this a
reproducible pipeline:

- :mod:`corread.stimuli`: word images and byte-reproducible word sets;
- :mod:`corread.training`: pre-literate, illiterate and literate models, one at a time or as a family;
- :mod:`corread.lesion`: word selectivity of the dense-layer units, lesions of them and the recovery;
- :mod:`corread.models`: the networks;
- :mod:`corread.hub`: sharing models and word sets through the Hugging Face Hub;
- :mod:`corread.imagenet`: preparing ImageNet.
"""

from __future__ import annotations

import logging

from ._provenance import package_version
from .imagenet import prepare_imagenet
from .lesion import LesionInstance
from .stimuli import generate_image, generate_training_set, regenerate_training_set, report_wordset_status
from .training import FamilyTrainingInstance, TrainingInstance

__version__ = package_version()

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'FamilyTrainingInstance',
    'LesionInstance',
    'TrainingInstance',
    '__version__',
    'generate_image',
    'generate_training_set',
    'prepare_imagenet',
    'regenerate_training_set',
    'report_wordset_status',
]
