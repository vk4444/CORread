# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Small file, folder and time helpers shared across the package."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def check_name(name: str | None, what: str = 'name') -> None:
    """Raise ``ValueError`` unless ``name`` is a plain folder name (no separators, not ``.``/``..``)."""
    if not name or name in ('.', '..') or any(s in name for s in ('/', '\\', os.sep)):
        raise ValueError(what + ' must be a plain folder name, got: ' + repr(name))


def write_json(path: str, data: Any) -> None:
    """Write ``data`` as JSON through a temporary file, so an interrupted write never leaves half a file."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def now_iso(t: float | None = None) -> str:
    """A time (default: now) as local ISO 8601 with its UTC offset, to the second."""
    return datetime.fromtimestamp(time.time() if t is None else t).astimezone().isoformat(timespec='seconds')


def ordered_map(fn: Callable[[T], R], items: Iterable[T], workers: int, ahead: int) -> Iterator[R]:
    """Map ``fn`` over ``items`` on a thread pool and yield the results in order.

    At most ``ahead`` results are pending at a time, so memory stays bounded.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque = deque()
        for item in items:
            pending.append(executor.submit(fn, item))
            if len(pending) >= ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def rename_dir(src: str, dst: str, attempts: int = 30) -> None:
    """Rename a folder, retrying while another program briefly holds a file in it open (Windows)."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(min(0.1 * 2**i, 2))


def model_parts(model: str) -> tuple[str, str]:
    """``(group, name)`` of a model given as ``'<group>/<name>'``, its folder under the models folder."""
    parts = model.replace('\\', '/').strip('/').split('/')
    if len(parts) != 2:
        raise ValueError(
            "model must be '<group>/<name>' as in the models folder, e.g. 'pre/pre_z_49', got: " + repr(model)
        )
    for part in parts:
        check_name(part, 'each part of model')
    return parts[0], parts[1]
