# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Model repos: one per model folder, claimed when a run starts and updated after every epoch."""

from __future__ import annotations

import json
import logging
import os
import shutil

from .._provenance import sha256_file
from ._cards import claim_card, model_card
from ._client import SIDECAR, connect, model_repo, remote_card, remote_file, sidecar, url

logger = logging.getLogger(__name__)


def _model_files(folder: str, complete: bool) -> list[str]:
    """The files of a model folder that go to the Hub: all but checkpoints once complete, else only the metadata."""
    if not complete:
        return ['metadata.json']
    files = []
    for root, dirs, names in os.walk(folder):
        if root == folder:
            dirs[:] = [d for d in dirs if d != 'checkpoints']
        for n in names:
            if n != SIDECAR and not n.endswith(('.part', '.tmp')):
                files.append(os.path.relpath(os.path.join(root, n), folder).replace(os.sep, '/'))
    return sorted(files)


def publish_model(
    repo_id: str, folder: str, group: str, message: str, status: str | None = None, delete_others: bool = False
):
    """Commit a model folder to its repo in one commit, so the repo never has metadata naming a file it lacks.

    ``status`` only changes the card (e.g. ``'failed'``, which ``metadata.json`` does not record).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    api, org, _ = connect(write=True)
    with open(os.path.join(folder, 'metadata.json'), 'rb') as f:
        raw = f.read()
    meta = json.loads(raw)
    files = _model_files(folder, meta.get('status') == 'complete')
    ops = [
        CommitOperationAdd('metadata.json', raw),
        CommitOperationAdd('README.md', model_card(api, org, group, meta, status).encode('utf-8')),
    ]
    ops += [CommitOperationAdd(p, os.path.join(folder, *p.split('/'))) for p in files if p != 'metadata.json']
    if delete_others:
        keep = set(files) | {'README.md', '.gitattributes'}
        ops += [CommitOperationDelete(p) for p in api.list_repo_files(repo_id) if p not in keep]
    return api.create_commit(repo_id, ops, commit_message=message)


def _taken(repo_id: str, card: dict) -> str:
    return (
        repo_id
        + ' on the Hub holds another run (status '
        + str(card.get('status'))
        + ', run '
        + str(card.get('run_id'))
        + '); choose another name, or delete that repo at '
        + url(repo_id, 'model')
        + '/settings'
    )


def _create_model_repo(api, repo_id: str, private: bool) -> None:
    """Create a model's repo, or raise ``FileExistsError`` if another run just did."""
    from huggingface_hub.errors import HfHubHTTPError

    try:
        api.create_repo(repo_id, private=private)
    except HfHubHTTPError as e:
        if getattr(e.response, 'status_code', None) == 409:
            raise FileExistsError(
                repo_id + ' was just created on the Hub by another run; choose another name'
            ) from None
        raise


def start_model_export(group: str, name: str, run_id: str) -> str:
    """Claim the repo of a run with ``export=True``, before its datasets load; returns the repo id.

    A new run creates its repo, which claims the name; a resumed run must find its own repo (same
    ``run_id``) or none.
    """
    from huggingface_hub import CommitOperationAdd

    api, org, private = connect(write=True)
    repo_id = model_repo(org, group, name)
    card = remote_card(api, repo_id, 'model')
    if card is None:
        _create_model_repo(api, repo_id, private)
        api.create_commit(
            repo_id,
            [CommitOperationAdd('README.md', claim_card(group, name, run_id).encode('utf-8'))],
            commit_message='start run ' + run_id,
        )
    elif card.get('run_id') != run_id:
        raise FileExistsError(_taken(repo_id, card))
    return repo_id


def discard_model_export(repo_id: str, run_id: str) -> None:
    """Delete the repo of a run that left nothing behind, if the repo is still that run's."""
    api, _, _ = connect(write=True)
    card = remote_card(api, repo_id, 'model')
    if card is not None and card.get('run_id') == run_id:
        api.delete_repo(repo_id)
        logger.info('deleted %s from the Hub', repo_id)


def export_model_folder(group: str, name: str, folder: str, overwrite: bool = False) -> str:
    """Upload ``models/<group>/<name>`` as it is now; returns the repo id."""
    api, org, private = connect(write=True)
    repo_id = model_repo(org, group, name)
    with open(os.path.join(folder, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    card = remote_card(api, repo_id, 'model')
    replacing = card is not None and card.get('run_id') != meta.get('run_id')
    if replacing and not overwrite:
        raise FileExistsError(_taken(repo_id, card) + ' (pass overwrite=True to replace it)')
    if card is None:
        _create_model_repo(api, repo_id, private)
    publish_model(
        repo_id,
        folder,
        group,
        'export: ' + str(meta.get('status')) + ', ' + str(meta.get('epochs_completed')) + ' epoch(s)',
        delete_others=replacing,
    )
    return repo_id


def download_model(group: str, name: str, part: str) -> str:
    """Download a finished model into ``part`` and check its file against its metadata; returns the repo id."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api, org, _ = connect()
    repo_id = model_repo(org, group, name)
    try:
        revision = api.model_info(repo_id).sha
    except RepositoryNotFoundError:
        raise FileNotFoundError('Not on the Hub (or not shared with you): ' + url(repo_id, 'model')) from None
    raw = remote_file(api, repo_id, 'metadata.json', 'model', revision)
    if raw is None:
        raise FileNotFoundError(repo_id + ' has no metadata.json yet; its run is still starting')
    meta = json.loads(raw)
    if meta.get('status') != 'complete':
        raise RuntimeError(
            repo_id
            + ' is not complete (status: '
            + str(meta.get('status'))
            + ', '
            + str(meta.get('epochs_completed'))
            + ' epoch(s)); only its metadata.json is on the '
            'Hub until training finishes: ' + url(repo_id, 'model')
        )
    if meta.get('name') != name:
        raise RuntimeError(repo_id + ' holds the model ' + repr(meta.get('name')) + ', not ' + repr(name))
    snapshot_download(repo_id, revision=revision, local_dir=part, ignore_patterns=['README.md', '.gitattributes'])
    # download bookkeeping that snapshot_download keeps inside local_dir
    shutil.rmtree(os.path.join(part, '.cache'), ignore_errors=True)
    model_file = os.path.join(part, name + '.pth.tar')
    if not os.path.isfile(model_file) or sha256_file(model_file) != meta.get('model_file_sha256'):
        raise RuntimeError(
            name + '.pth.tar from ' + repo_id + ' does not match the model_file_sha256 in its metadata.json'
        )
    with open(os.path.join(part, SIDECAR), 'w', encoding='utf-8') as f:
        json.dump(sidecar(repo_id, 'model', revision), f, indent=2)
    return repo_id
