# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Word set recipes, and the fonts and wordlists they are drawn from (the private ``<org>/assets`` repo)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

from .._io import read_json
from .._provenance import sha256_file
from ..paths import get_path
from ._cards import assets_card, wordset_card
from ._client import (
    ASSETS_REPO,
    RECIPE_FILES,
    REVISION,
    SIDECAR,
    connect,
    remote_card,
    remote_file,
    sidecar,
    url,
    valid_repo_id,
    wordset_repo,
)

# the files of the assets repo a recipe may name: one level deep, plain names
ASSET_PATH = re.compile(r'^(fonts/[^/\\]+\.ttf|wordlists/[^/\\]+\.txt)$')


def _check_asset_path(path_in_repo: str) -> str:
    name = path_in_repo.split('/', 1)[-1]
    if not ASSET_PATH.match(path_in_repo) or name in ('.', '..') or name.startswith('.'):
        raise ValueError("Unexpected file in a word set's sources.json: " + repr(path_in_repo))
    return path_in_repo


def sync_assets(files: dict[str, str], overwrite: bool = False) -> str:
    """Make ``<org>/assets`` hold ``files`` (``{path in repo: local path}``); returns the revision holding them all.

    Files it lacks are uploaded; one it has with different content is an error unless ``overwrite``.
    The repo is always private, since fonts are often licensed for your own use only.
    """
    from huggingface_hub import CommitOperationAdd

    api, org, _ = connect(write=True)
    repo_id = org + '/' + ASSETS_REPO
    ops = []
    if remote_card(api, repo_id, 'dataset') is None:
        api.create_repo(repo_id, repo_type='dataset', private=True)
        ops.append(CommitOperationAdd('README.md', assets_card(org).encode('utf-8')))
    head = api.dataset_info(repo_id).sha
    differ = []
    for path_in_repo, local in sorted(files.items()):
        remote = remote_file(api, repo_id, path_in_repo, 'dataset', head)
        if remote is not None and hashlib.sha256(remote).hexdigest() == sha256_file(local):
            continue
        if remote is not None and not overwrite:
            differ.append(path_in_repo)
        else:
            ops.append(CommitOperationAdd(path_in_repo, local))
    if differ:
        raise FileExistsError(
            repo_id
            + ' has different files under these names: '
            + ', '.join(differ)
            + '; rename the local files, or replace those on the Hub with '
            'hub.export_assets(overwrite=True) (word set recipes keep pointing to the old '
            'versions, so they still regenerate)'
        )
    if not ops:
        return head
    added = [op.path_in_repo for op in ops if op.path_in_repo != 'README.md']
    return api.create_commit(
        repo_id, ops, commit_message='add ' + ', '.join(added), parent_commit=head, repo_type='dataset'
    ).oid


def import_assets_into(fontsdir: str, wordsdir: str, overwrite: bool = False) -> tuple[str, list[str]]:
    """Download every font and wordlist of ``<org>/assets``; returns the repo and the files copied."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api, org, _ = connect()
    repo_id = org + '/' + ASSETS_REPO
    try:
        revision = api.dataset_info(repo_id).sha
    except RepositoryNotFoundError:
        raise FileNotFoundError('Not on the Hub (or not shared with you): ' + url(repo_id, 'dataset')) from None
    snapshot = snapshot_download(
        repo_id, repo_type='dataset', revision=revision, allow_patterns=['fonts/*', 'wordlists/*']
    )
    copies, differ = [], []
    for sub, dest_dir in (('fonts', fontsdir), ('wordlists', wordsdir)):
        src_dir = os.path.join(snapshot, sub)
        for n in sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []:
            src, dest = os.path.join(src_dir, n), os.path.join(dest_dir, n)
            if os.path.isfile(dest) and sha256_file(dest) == sha256_file(src):
                continue
            (differ if os.path.exists(dest) and not overwrite else copies).append((src, dest))
    if differ:
        raise FileExistsError(
            'These local files differ from '
            + repo_id
            + ': '
            + ', '.join(d for _, d in differ)
            + ' (pass overwrite=True to replace them)'
        )
    for src, dest in copies:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
    return repo_id, [d for _, d in copies]


def publish_wordset(folder: str, inputs: dict[str, str], overwrite: bool = False) -> str:
    """Upload a word set's recipe; its fonts and wordlist (``inputs``) go to ``<org>/assets``. Returns the repo."""
    from huggingface_hub import CommitOperationAdd

    api, org, private = connect(write=True)
    with open(os.path.join(folder, 'metadata.json'), 'rb') as f:
        raw = f.read()
    meta = json.loads(raw)
    repo_id = wordset_repo(org, meta['name'])
    remote = remote_file(api, repo_id, 'metadata.json', 'dataset')
    if remote is not None and remote != raw and not overwrite:
        raise FileExistsError(
            repo_id + ' on the Hub holds a different word set recipe (pass overwrite=True to '
            "replace it; its old revisions stay in the repo's history)"
        )
    revision = sync_assets(inputs)
    sources = {
        'assets': {
            'repo_id': org + '/' + ASSETS_REPO,
            'revision': revision,
            'files': {p: sha256_file(local) for p, local in sorted(inputs.items())},
        }
    }
    checksums = read_json(os.path.join(folder, 'checksums.json'))
    if remote_card(api, repo_id, 'dataset') is None:
        api.create_repo(repo_id, repo_type='dataset', private=private)
    ops = [
        CommitOperationAdd('metadata.json', raw),
        CommitOperationAdd('images.csv', os.path.join(folder, 'images.csv')),
        CommitOperationAdd('checksums.json', os.path.join(folder, 'checksums.json')),
        CommitOperationAdd('sources.json', (json.dumps(sources, indent=2) + '\n').encode('utf-8')),
        CommitOperationAdd('README.md', wordset_card(meta, checksums, sources).encode('utf-8')),
    ]
    api.create_commit(repo_id, ops, commit_message='recipe of ' + meta['name'], repo_type='dataset')
    return repo_id


def export_wordset_folder(
    folder: str, overwrite: bool = False, fontsdir: str | None = None, wordsdir: str | None = None
) -> str:
    """Upload the recipe of the word set in ``folder``; ``fontsdir`` and ``wordsdir`` hold its inputs."""
    from ..stimuli import _recipe_inputs

    meta_path = os.path.join(folder, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError('No word set: ' + meta_path + ' does not exist')
    meta = read_json(meta_path)
    if meta.get('status') != 'complete':
        raise RuntimeError('Word set ' + folder + ' is not complete (status: ' + str(meta.get('status')) + ')')
    if meta.get('name') != os.path.basename(os.path.normpath(folder)):
        raise ValueError(folder + ' holds the word set ' + repr(meta.get('name')) + '; rename the folder back')
    if not os.path.isfile(os.path.join(folder, 'checksums.json')):
        raise FileNotFoundError('Word set ' + folder + ' has no checksums.json')
    fonts, words_path = _recipe_inputs(
        meta,
        fontsdir if fontsdir is not None else get_path('fonts'),
        wordsdir if wordsdir is not None else get_path('wordlists'),
    )
    inputs = {'fonts/' + fontname + '.ttf': path for fontname, path in fonts.items()}
    inputs['wordlists/' + meta['arguments']['wordlist'] + '.txt'] = words_path
    return publish_wordset(folder, inputs, overwrite=overwrite)


def download_recipe(name: str, recipe_dir: str) -> str:
    """Download the recipe of word set ``name`` into ``recipe_dir``, with the fonts and wordlist it pins.

    The fonts and wordlist go under ``recipe_dir/fonts`` and ``recipe_dir/wordlists``, each checked
    against the sha256 in ``sources.json``. Returns the repo id.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api, org, _ = connect()
    repo_id = wordset_repo(org, name)
    try:
        revision = api.dataset_info(repo_id).sha
    except RepositoryNotFoundError:
        raise FileNotFoundError('Not on the Hub (or not shared with you): ' + url(repo_id, 'dataset')) from None
    snapshot = snapshot_download(repo_id, repo_type='dataset', revision=revision, allow_patterns=list(RECIPE_FILES))
    missing = [f for f in RECIPE_FILES if not os.path.isfile(os.path.join(snapshot, f))]
    if missing:
        raise FileNotFoundError(repo_id + ' is not a complete word set recipe; it lacks ' + ', '.join(missing))
    os.makedirs(recipe_dir)
    for f in RECIPE_FILES:
        shutil.copyfile(os.path.join(snapshot, f), os.path.join(recipe_dir, f))
    assets = read_json(os.path.join(recipe_dir, 'sources.json'))['assets']
    valid_repo_id(assets['repo_id'], 'The assets repo of ' + repo_id)
    if not isinstance(assets.get('revision'), str) or not REVISION.match(assets['revision']):
        raise ValueError('sources.json of ' + repo_id + ' does not pin a commit of ' + assets['repo_id'])
    files = {_check_asset_path(p): sha for p, sha in assets['files'].items()}
    inputs = snapshot_download(
        assets['repo_id'], repo_type='dataset', revision=assets['revision'], allow_patterns=list(files)
    )
    for path_in_repo, sha in files.items():
        dest = os.path.join(recipe_dir, *path_in_repo.split('/'))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(os.path.join(inputs, *path_in_repo.split('/')), dest)
        if sha256_file(dest) != sha:
            raise RuntimeError(path_in_repo + ' from ' + assets['repo_id'] + ' does not match sources.json')
    with open(os.path.join(recipe_dir, SIDECAR), 'w', encoding='utf-8') as f:
        json.dump(sidecar(repo_id, 'dataset', revision, assets=assets), f, indent=2)
    return repo_id
