# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""A family of models: the pre-literate model and every model trained from it."""

from __future__ import annotations

import logging
from typing import Any

from .instance import TrainingInstance

logger = logging.getLogger(__name__)

FAMILY_MODES = ('pre', 'illit', 'lit_no_bias', 'lit_bias')
"""The order a family trains in: the pre-literate model first, as the others start from it."""


class FamilyTrainingInstance:
    """Train a pre-literate model, then the illiterate, unbiased and biased literate models from it.

    Takes the arguments of :class:`TrainingInstance` except ``mode``; ``pre_name`` names the
    pre-literate model and ``name`` the others. A model already trained with these arguments is skipped,
    so a family can be run again after an interruption, or for another word set, reusing its
    pre-literate and illiterate models.

    Attributes
    ----------
    instances : dict
        The :class:`TrainingInstance` of each mode, in training order.
    """

    def __init__(self, name: str | None = None, pre_name: str | None = None, **kwargs: Any):
        if 'mode' in kwargs:
            raise TypeError('FamilyTrainingInstance trains every mode, so it takes no mode')
        self.instances = {'pre': TrainingInstance(mode='pre', name=pre_name, **kwargs)}
        for mode in FAMILY_MODES[1:]:
            self.instances[mode] = TrainingInstance(mode=mode, name=name, pre_name=pre_name, **kwargs)

    def train(self) -> None:
        """Train the models in order; stops at the first that fails or is interrupted, since later ones need it."""
        for instance in self.instances.values():
            if instance.is_complete():
                logger.info('already trained, skipping %s', instance.model_dir)
                continue
            instance.train()
