# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2021 T. Hannagan, A. Agrawal, L. Cohen, S. Dehaene (Origins-of-VWFA)
# Copyright (C) 2026 Vojtěch Kubala
#
# The image and word datasets follow https://github.com/THANNAGA/Origins-of-VWFA (clean_train.py),
# via Dr. Olga Solaja's adaptation. Modified by Vojtěch Kubala, 2026: fixed validation views and
# word labels offset by a picklable transform.
"""The image and word datasets: ImageNet-style folders (``train/<class>/``, ``val/<class>/``) as tensors."""

from __future__ import annotations

import os

import torchvision
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_tensor() -> list:
    return [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]


def image_dataset(
    data_path: str, folder: str = 'train', augment: bool | None = None
) -> torchvision.datasets.ImageFolder:
    """The images of ``data_path/folder``, one class per subfolder.

    With ``augment`` (by default for the train folder) each image is a random resized 224 px crop, a
    new one every epoch; otherwise it is the standard ImageNet evaluation view (resize to 256, central
    224 px crop), the same every epoch, so validation scores are comparable.
    """
    if augment is None:
        augment = folder == 'train'
    if augment:
        view = [transforms.RandomResizedCrop(224)]
    else:
        view = [transforms.Resize(256), transforms.CenterCrop(224)]
    return torchvision.datasets.ImageFolder(os.path.join(data_path, folder), transforms.Compose(view + _to_tensor()))


class OffsetLabels:
    """Add a fixed offset to a class index, so word classes follow the image classes."""

    def __init__(self, offset: int):
        self.offset = offset

    def __call__(self, target: int) -> int:
        return target + self.offset

    def __repr__(self) -> str:
        return 'OffsetLabels(' + str(self.offset) + ')'


def word_dataset(data_path: str, folder: str = 'train', label_offset: int = 1000) -> torchvision.datasets.ImageFolder:
    """The word images of a word set, labelled ``label_offset`` + the word's index.

    Training images are random crops of 90–100 % of the image, kept square; validation images are the
    whole image resized to 224 × 224, so no letters are cut off and the view is the same every epoch.
    """
    if folder == 'train':
        view = [transforms.RandomResizedCrop(224, scale=(0.9, 1), ratio=(1, 1))]
    else:
        view = [transforms.Resize((224, 224))]
    return torchvision.datasets.ImageFolder(
        os.path.join(data_path, folder),
        transforms.Compose(view + _to_tensor()),
        target_transform=OffsetLabels(label_offset),
    )
