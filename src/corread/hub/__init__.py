# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Share trained models and word sets through a Hugging Face organisation of your own.

Set the organisation (or your user name) in ``paths.toml``, and sign in once with ``hf auth login``
(or set ``HF_TOKEN``; the token never goes in ``paths.toml``)::

    [hub]
    org = "my-lab"
    private = true    # visibility of the repos created from here on

The organisation holds three kinds of repo:

- ``<org>/<group>.<name>`` (model): the model in ``models/<group>/<name>``: its ``metadata.json``
  from the first epoch on, and once complete the model file, scores and log; never the checkpoints.
- ``<org>/<word set>`` (dataset): the word set's recipe, never its images; :func:`import_wordset`
  draws them again, byte for byte.
- ``<org>/assets`` (dataset, always private): the fonts and wordlists the recipes are drawn from.

Every repo gets a generated card; a model's card links the pre-literate model it started from and the
word set it was trained on, so the Hub shows the lineage of every model. See ``docs/hub.md``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

from .._io import check_name, model_parts, rename_dir
from ..paths import get_path
from ._client import CARD_KEY, connect, url
from ._models import download_model, export_model_folder
from ._wordsets import download_recipe, export_wordset_folder, import_assets_into, sync_assets

logger = logging.getLogger(__name__)

__all__ = [
    'export_assets',
    'export_model',
    'export_wordset',
    'import_assets',
    'import_model',
    'import_wordset',
    'list_models',
    'list_wordsets',
]


def export_model(model: str, overwrite: bool = False) -> str:
    """Upload the model folder ``models/<model>`` (``'<group>/<name>'``, e.g. ``'pre/pre_z_49'``).

    Uploads ``metadata.json``, and once training is complete also the model file, scores and log. A
    repo holding another run is an error unless ``overwrite``. ``TrainingInstance(export=True)`` does
    this as it trains. Returns the repo id.
    """
    group, name = model_parts(model)
    folder = os.path.join(get_path('models'), group, name)
    meta_path = os.path.join(folder, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError('Not a model folder (no metadata.json): ' + folder)
    with open(meta_path, encoding='utf-8') as f:
        if json.load(f).get('mode') == 'lesion':
            raise NotImplementedError('Exporting lesion runs to the Hub is not supported: ' + folder)
    repo_id = export_model_folder(group, name, folder, overwrite=overwrite)
    logger.info('Exported %s to %s', folder, url(repo_id, 'model'))
    return repo_id


def import_model(model: str) -> str:
    """Download the finished model ``<org>/<group>.<name>`` into ``models/<model>``; returns the folder.

    The model file is checked against the hash in its metadata, and the folder only appears once
    everything is in place.
    """
    group, name = model_parts(model)
    folder = os.path.join(get_path('models'), group, name)
    if os.path.exists(folder):
        raise FileExistsError('Already exists: ' + folder + ' (delete it or rename it first)')
    part = folder + '.part'
    shutil.rmtree(part, ignore_errors=True)
    try:
        repo_id = download_model(group, name, part)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    rename_dir(part, folder)
    logger.info('Imported %s to %s', url(repo_id, 'model'), folder)
    return folder


def export_wordset(name: str, overwrite: bool = False) -> str:
    """Upload the recipe of the word set ``wordsets/<name>``, and its fonts and wordlist to ``<org>/assets``.

    The recipe pins the revision of ``<org>/assets`` that holds its inputs, so a font replaced there
    later does not change what the recipe regenerates. Returns the repo id.
    """
    check_name(name)
    repo_id = export_wordset_folder(os.path.join(get_path('wordsets'), name), overwrite=overwrite)
    logger.info('Exported the recipe of %s to %s', name, url(repo_id, 'dataset'))
    return repo_id


def import_wordset(name: str, workers: int = 1, allow_mismatch: bool = False) -> str:
    """Download the recipe of word set ``name`` and draw its images again into ``wordsets/<name>``.

    Every word folder is checked against the recipe's checksums; see
    :func:`corread.stimuli.regenerate_training_set`. Returns the folder.
    """
    from ..stimuli import regenerate_training_set

    check_name(name)
    wordsets = get_path('wordsets')
    folder = os.path.join(wordsets, name)
    if os.path.exists(folder):
        raise FileExistsError('Already exists: ' + folder + ' (delete it or rename it first)')
    recipe = os.path.join(wordsets, name + '.recipe')
    shutil.rmtree(recipe, ignore_errors=True)
    try:
        download_recipe(name, recipe)
        return regenerate_training_set(
            name,
            recipe=recipe,
            workers=workers,
            wordsetsdir=wordsets,
            fontsdir=os.path.join(recipe, 'fonts'),
            wordsdir=os.path.join(recipe, 'wordlists'),
            allow_mismatch=allow_mismatch,
        )
    finally:
        shutil.rmtree(recipe, ignore_errors=True)


def export_assets(overwrite: bool = False) -> str:
    """Upload every font (``fonts/*.ttf``) and wordlist (``wordlists/*.txt``) to the private ``<org>/assets``.

    A file already there with different content is an error unless ``overwrite`` (word set recipes keep
    pointing to the old one). Returns the revision that holds them all.
    """
    files = {}
    for sub, folder, ext in (('fonts', get_path('fonts'), '.ttf'), ('wordlists', get_path('wordlists'), '.txt')):
        for n in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if n.endswith(ext) and os.path.isfile(os.path.join(folder, n)):
                files[sub + '/' + n] = os.path.join(folder, n)
    if not files:
        raise FileNotFoundError(
            'No fonts or wordlists to export in ' + get_path('fonts') + ' or ' + get_path('wordlists')
        )
    revision = sync_assets(files, overwrite=overwrite)
    logger.info('Exported %d fonts and wordlists; the Hub has them all at revision %s', len(files), revision)
    return revision


def import_assets(overwrite: bool = False) -> list[str]:
    """Download every font and wordlist of ``<org>/assets`` into your fonts and wordlists folders.

    A local file with different content is an error unless ``overwrite``. Returns the files copied.
    """
    repo_id, copied = import_assets_into(get_path('fonts'), get_path('wordlists'), overwrite=overwrite)
    logger.info('Imported %d new or changed files from %s', len(copied), repo_id)
    return copied


def _list(kind: str) -> list[dict[str, Any]]:
    api, org, _ = connect()
    if kind == 'model':
        infos = api.list_models(author=org, cardData=True)
    else:
        infos = api.list_datasets(author=org, expand=['cardData'])
    found = []
    for info in infos:
        card = (info.card_data.to_dict() if info.card_data is not None else {}).get(CARD_KEY) or {}
        if card.get('kind') == kind:
            found.append(dict(card, repo=info.id, url=url(info.id, 'model' if kind == 'model' else 'dataset')))
    return sorted(found, key=lambda c: str(c.get('model') or c.get('name')))


def list_models() -> list[dict[str, Any]]:
    """The models in the organisation: ``model`` (``'<group>/<name>'``), repo, url, status, mode, accuracies, ..."""
    return _list('model')


def list_wordsets() -> list[dict[str, Any]]:
    """The word set recipes in the organisation: name, repo, url, wordlist, fonts, number of images, ..."""
    return _list('wordset')
