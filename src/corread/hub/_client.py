# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""The connection to the Hugging Face Hub, repo names, and the small reads every operation shares."""

from __future__ import annotations

import json
import os
import re
from functools import cache
from typing import Any

from .._io import now_iso
from ..paths import get_section

ASSETS_REPO = 'assets'
SIDECAR = '.hub.json'
"""File in an imported model or word set folder recording the repo and revision it came from."""
PRE_GROUP = 'pre'
IMAGENET_DATASET = 'ILSVRC/imagenet-1k'
RECIPE_FILES = ('metadata.json', 'images.csv', 'checksums.json', 'sources.json')
CARD_KEY = 'corread'
"""The section of a repo card that :func:`corread.hub.list_models` and ``list_wordsets`` read."""
TAG = 'corread'
REVISION = re.compile(r'^[0-9a-f]{40}$')


def settings() -> tuple[str, bool]:
    """``(org, private)`` from the ``[hub]`` table of ``paths.toml``."""
    try:
        section = get_section('hub')
    except KeyError:
        raise ValueError(
            'Sharing through the Hugging Face Hub needs a [hub] section with '
            'org = "<organisation or user>" in paths.toml'
        ) from None
    org = section.get('org')
    if not isinstance(org, str) or not org or '/' in org:
        raise ValueError(
            '[hub] org in paths.toml must be the name of a Hugging Face organisation (or user), got: ' + repr(org)
        )
    private = section.get('private', True)
    if not isinstance(private, bool):
        raise ValueError('[hub] private in paths.toml must be true or false, got: ' + repr(private))
    return org, private


@cache
def connect(write: bool = False):
    """``(HfApi client, org, private)``, checked once per process before anything long starts.

    Checks that you are signed in and, with ``write``, that you may change repos in the organisation.
    The token comes from ``hf auth login`` or ``HF_TOKEN``, never from ``paths.toml``.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError:
        raise ImportError("Sharing through the Hub needs huggingface_hub: pip install 'corread[hub]'") from None
    org, private = settings()
    api = HfApi()
    try:
        me = api.whoami()
    except LocalTokenNotFoundError:
        raise RuntimeError('Not signed in to Hugging Face: run `hf auth login` (or set HF_TOKEN)') from None
    except Exception as e:
        raise RuntimeError(
            'Cannot reach Hugging Face as a signed-in user ('
            + str(e).strip().splitlines()[0]
            + '); check the connection, or sign in again with `hf auth login`'
        ) from None
    if write:
        if (me.get('auth') or {}).get('accessToken', {}).get('role') == 'read':
            raise RuntimeError('The Hugging Face token is read-only; sign in with a write token (`hf auth login`)')
        if org != me.get('name'):
            role = next((o.get('roleInOrg') for o in me.get('orgs', []) if o.get('name') == org), None)
            if role is None:
                raise RuntimeError(str(me.get('name')) + ' is not a member of the Hugging Face organisation ' + org)
            if role == 'read':
                raise RuntimeError(
                    str(me.get('name')) + ' has the read role in ' + org + '; uploading needs the write role'
                )
    return api, org, private


def valid_repo_id(repo_id: str, what: str) -> str:
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(repo_id)
    except HFValidationError as e:
        raise ValueError(what + ' cannot be a Hugging Face repo: ' + str(e)) from None
    return repo_id


def model_repo(org: str, group: str, name: str) -> str:
    """The repo of ``models/<group>/<name>``; the group is part of it, since literate models of every word
    set share their names. A dot joins them, as the Hub forbids ``--`` in repo names."""
    return valid_repo_id(org + '/' + group + '.' + name, 'Model ' + group + '/' + name)


def wordset_repo(org: str, name: str) -> str:
    if name == ASSETS_REPO:
        raise ValueError(
            'A word set named '
            + repr(name)
            + ' cannot go to the Hub; '
            + org
            + '/'
            + ASSETS_REPO
            + ' holds the fonts and wordlists'
        )
    return valid_repo_id(org + '/' + name, 'Word set ' + name)


def url(repo_id: str, repo_type: str) -> str:
    return 'https://huggingface.co/' + ('datasets/' if repo_type == 'dataset' else '') + repo_id


def remote_card(api, repo_id: str, repo_type: str) -> dict[str, Any] | None:
    """The CORread section of a repo's card: ``{}`` when the card has none, ``None`` when there is no repo."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return None
    data = info.card_data.to_dict() if info.card_data is not None else {}
    return data.get(CARD_KEY) or {}


def remote_file(api, repo_id: str, filename: str, repo_type: str, revision: str | None = None) -> bytes | None:
    """A file of a repo, or ``None`` when the repo or the file does not exist."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = api.hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    with open(path, 'rb') as f:
        return f.read()


# repos known to exist; cards are rebuilt after every epoch, and a base model or word set does not disappear
_existing: set[tuple[str, str]] = set()


def exists(api, repo_id: str, repo_type: str) -> bool:
    if (repo_id, repo_type) not in _existing and api.repo_exists(repo_id, repo_type=repo_type):
        _existing.add((repo_id, repo_type))
    return (repo_id, repo_type) in _existing


def sidecar(repo_id: str, repo_type: str, revision: str, **extra: Any) -> dict[str, Any]:
    return dict(
        {
            'repo_id': repo_id,
            'repo_type': repo_type,
            'revision': revision,
            'url': url(repo_id, repo_type),
            'imported_at': now_iso(),
        },
        **extra,
    )


def read_sidecar(folder: str) -> dict[str, Any] | None:
    """Where an imported model or word set folder came from (its ``.hub.json``), or ``None``."""
    path = os.path.join(folder, SIDECAR)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def yaml_card(body: str, dataset: bool = False, **data: Any) -> str:
    """A repo card: the YAML metadata block, then the Markdown body."""
    from huggingface_hub import DatasetCardData, ModelCardData

    card_data = (DatasetCardData if dataset else ModelCardData)(**data)
    return '---\n' + card_data.to_yaml() + '\n---\n\n' + body
