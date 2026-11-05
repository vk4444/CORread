# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Lesions of the dense layer of a literate CORnet-Z, the recovery from them, and word selectivity."""

from __future__ import annotations

from .instance import LesionInstance
from .units import check_units, lesion, selective, selectivity, word_channel

__all__ = ['LesionInstance', 'check_units', 'lesion', 'selective', 'selectivity', 'word_channel']
