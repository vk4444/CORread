# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Where CORread finds its inputs and writes its outputs: the ``paths.toml`` configuration file.

The file is the one named by the ``CORREAD_PATHS`` environment variable, or else the first
``paths.toml`` in the working directory or one of its parents. Its top-level entries are folders,
relative to the file; its tables (``[hub]``, ``[report]``) hold the settings of optional features.
See ``examples/paths.example.toml``.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

CONFIG_NAME = 'paths.toml'
ENV_VAR = 'CORREAD_PATHS'


def find_config() -> str:
    """The absolute path of the configuration file in use."""
    env = os.environ.get(ENV_VAR)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(ENV_VAR + ' points to a missing file: ' + env)
        return os.path.abspath(env)
    folder = os.getcwd()
    while True:
        candidate = os.path.join(folder, CONFIG_NAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    raise FileNotFoundError(
        'No ' + CONFIG_NAME + ' in ' + os.getcwd() + ' or its parents; create one (see '
        'examples/paths.example.toml) or set ' + ENV_VAR + ' to its path.'
    )


@cache
def _read() -> tuple[str, dict[str, Any]]:
    config = find_config()
    with open(config, 'rb') as f:
        return config, tomllib.load(f)


def clear_cache() -> None:
    """Forget the configuration read so far, e.g. after changing ``CORREAD_PATHS``."""
    _read.cache_clear()


def get_path(name: str) -> str:
    """The absolute path of the folder ``name`` (``fonts``, ``wordlists``, ``imagesets``, ``wordsets``, ``models``)."""
    config, entries = _read()
    root = os.path.dirname(config)
    folders = {k: v for k, v in entries.items() if not isinstance(v, dict)}
    if name not in folders:
        raise KeyError("'" + name + "' is not in " + config + '; it has: ' + ', '.join(sorted(folders)))
    return os.path.normpath(os.path.join(root, os.path.expanduser(folders[name])))


def get_section(name: str) -> dict[str, Any]:
    """The table ``[name]`` of the configuration, e.g. the ``[hub]`` settings."""
    config, entries = _read()
    if not isinstance(entries.get(name), dict):
        raise KeyError('No [' + name + '] section in ' + config)
    return entries[name]
