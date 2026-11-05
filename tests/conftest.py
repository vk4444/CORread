"""Shared fixtures: a complete, tiny CORread setup on synthetic data, trained on the CPU.

The images are 3 classes of seeded noise, the words 4 short words in DejaVu Sans; a trained family takes
about a minute. The network sizes stay at 1000 image and 1000 word classes, so the tiny data only uses a
few of them.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

import corread
from corread import paths

FONT = os.path.join(os.path.dirname(__file__), 'data', 'fonts', 'DejaVuSans.ttf')
WORDS = ['cat', 'house', 'tree', 'moon']
TINY = dict(seed=1234, max_epochs_pre=2, max_epochs_lit=2, batch_size=4, num_val_items=2, gpus=0, num_workers=0)

torch.use_deterministic_algorithms(True)
torch.set_num_threads(2)


def make_root(root) -> str:
    """Write a configuration and its data folders under ``root``; returns the ``paths.toml`` path."""
    root = str(root)
    os.makedirs(os.path.join(root, 'fonts'))
    shutil.copyfile(FONT, os.path.join(root, 'fonts', 'sans.ttf'))
    os.makedirs(os.path.join(root, 'wordlists'))
    with open(os.path.join(root, 'wordlists', 'words.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(WORDS) + '\n\nunused\n')
    rng = np.random.RandomState(0)
    for split, n in (('train', 6), ('val', 2)):
        for c in range(3):
            folder = os.path.join(root, 'imagesets', split, 'c%d' % c)
            os.makedirs(folder)
            for i in range(n):
                pixels = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
                pixels[..., c] = 255 - pixels[..., c] // 4
                Image.fromarray(pixels).save(os.path.join(folder, 'img%d.png' % i))
    for folder in ('wordsets', 'models'):
        os.makedirs(os.path.join(root, folder))
    config = os.path.join(root, 'paths.toml')
    with open(config, 'w', encoding='utf-8') as f:
        f.write(
            'fonts = "fonts"\nwordlists = "wordlists"\nimagesets = "imagesets"\nwordsets = "wordsets"\n'
            'models = "models"\n'
        )
    return config


def use_config(monkeypatch, config: str) -> None:
    monkeypatch.setenv(paths.ENV_VAR, config)
    paths.clear_cache()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh, empty setup for one test."""
    use_config(monkeypatch, make_root(tmp_path / 'root'))
    yield str(tmp_path / 'root')
    paths.clear_cache()


@pytest.fixture(scope='session')
def trained(tmp_path_factory):
    """A setup with a word set and a trained CORnet-Z family, shared by the tests that only read it."""
    base = tmp_path_factory.mktemp('trained')
    config = make_root(base)
    previous = os.environ.get(paths.ENV_VAR)
    os.environ[paths.ENV_VAR] = config
    paths.clear_cache()
    corread.generate_training_set(
        'words',
        'words',
        train_fonts=['sans'],
        val_fonts=['sans'],
        num_words=4,
        num_train=4,
        num_val=2,
        width=150,
        height=150,
        seed=7,
    )
    corread.FamilyTrainingInstance(model_choice='z', wordset='words', **TINY).train()
    yield str(base)
    if previous is None:
        os.environ.pop(paths.ENV_VAR, None)
    else:
        os.environ[paths.ENV_VAR] = previous
    paths.clear_cache()


@pytest.fixture
def in_trained(trained, monkeypatch):
    """Use the shared trained setup in a test."""
    use_config(monkeypatch, os.path.join(trained, 'paths.toml'))
    yield trained
    paths.clear_cache()
