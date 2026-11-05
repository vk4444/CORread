# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""What identifies the inputs and the code behind every output: file hashes and the package's version."""

from __future__ import annotations

import hashlib
import os
import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path: str) -> str:
    """The sha256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def package_version() -> str:
    try:
        return version('corread')
    except PackageNotFoundError:
        return '0+unknown'


def _code_sha256() -> str:
    """One hash over every source file of the package: their paths and contents, in sorted order."""
    files = []
    for root, dirs, names in os.walk(PACKAGE_DIR):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        files += [os.path.join(root, n) for n in names if n.endswith('.py')]
    h = hashlib.sha256()
    for path in sorted(files):
        rel = os.path.relpath(path, PACKAGE_DIR).replace(os.sep, '/')
        h.update((rel + ' ' + sha256_file(path) + '\n').encode('utf-8'))
    return h.hexdigest()


def _git(*args: str, timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', PACKAGE_DIR, *args], capture_output=True, text=True, timeout=timeout)


def _git_info() -> dict[str, Any] | None:
    """The git commit of the package's source and whether the source differs from it.

    ``None`` unless the package runs from a git checkout that tracks it (an installed copy inside some
    other repository's folder is not mistaken for that repository's code).
    """
    try:
        if _git('ls-files', '--error-unmatch', '__init__.py').returncode != 0:
            return None
        commit = _git('rev-parse', 'HEAD')
        if commit.returncode != 0:
            return None
        status = _git('status', '--porcelain', '--', '.', timeout=30)
        return {'commit': commit.stdout.strip(), 'source_modified_since_commit': bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


@cache
def code_provenance() -> dict[str, Any]:
    """The version, source hash and git state of the code this process runs.

    Computed once per process, at the first run that records it, so a long process keeps reporting the
    code it started with even if the checkout changes later.
    """
    return {'corread': package_version(), 'code_sha256': _code_sha256(), 'git': _git_info()}
