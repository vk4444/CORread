# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Training runs: pre-literate, illiterate and literate CORnets, one at a time or as a family."""

from __future__ import annotations

from .family import FAMILY_MODES, FamilyTrainingInstance
from .instance import LITERATE_MODES, MODES, TrainingInstance

__all__ = ['FAMILY_MODES', 'LITERATE_MODES', 'MODES', 'FamilyTrainingInstance', 'TrainingInstance']
