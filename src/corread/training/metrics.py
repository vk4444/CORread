# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Loss and accuracy of a pass over the data, overall and apart for the images and the words."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import torch
from torch import nn


def accuracy(out: torch.Tensor, labels: torch.Tensor) -> float:
    """The % of a batch whose largest output is its label."""
    pred = np.argmax(out.detach().cpu().numpy(), axis=1)
    return 100 * np.mean(pred == labels.cpu().numpy())


def batch_sums(pred: torch.Tensor, labels: torch.Tensor, img_classes: int) -> list[float]:
    """Summed cross-entropy, correct predictions and count of a batch's images, then of its words.

    Labels below ``img_classes`` are images. Computed from detached outputs, so it never affects training.
    """
    pred = pred.detach()
    losses = nn.functional.cross_entropy(pred, labels, reduction='none')
    correct = (pred.argmax(1) == labels).float()
    words = (labels >= img_classes).float()
    images = 1 - words
    return torch.stack(
        [
            (losses * images).sum(),
            (correct * images).sum(),
            images.sum(),
            (losses * words).sum(),
            (correct * words).sum(),
            words.sum(),
        ]
    ).tolist()


def means(prefix: str, sums) -> dict[str, float | None]:
    """Mean loss and accuracy (%) of a pass (``prefix`` ``'train'`` or ``'val'``) from its summed :func:`batch_sums`.

    The image and word values are ``None`` when the pass had none of them.
    """
    img_loss, img_correct, img_n, wrd_loss, wrd_correct, wrd_n = (float(s) for s in sums)
    return {
        prefix + '_loss': (img_loss + wrd_loss) / (img_n + wrd_n) if img_n + wrd_n else None,
        prefix + '_loss_images': img_loss / img_n if img_n else None,
        prefix + '_loss_words': wrd_loss / wrd_n if wrd_n else None,
        prefix + '_accuracy_images': 100 * img_correct / img_n if img_n else None,
        prefix + '_accuracy_words': 100 * wrd_correct / wrd_n if wrd_n else None,
    }


def duration_text(seconds: float) -> str:
    return str(timedelta(seconds=seconds))
