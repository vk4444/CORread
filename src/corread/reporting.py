# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Optional status reports of training runs, lesion runs and word set generation to a web API of your own.

Reporting is off unless a run is started with ``report=True``. It then POSTs JSON reports, as
specified in ``docs/reporting.md`` (and as a JSON Schema in ``docs/reporting.schema.json``), to the
URL in the ``[report]`` table of ``paths.toml``, with the token from the ``CORREAD_REPORT_TOKEN``
environment variable::

    [report]
    url = "https://example.org/api/runs"   # https:// required (http:// only for localhost)
    every = 60                              # seconds between progress reports

Reports are sent from a background thread and never stop a run.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import platform
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from ._io import now_iso
from .paths import get_section

logger = logging.getLogger(__name__)

SCHEMA = 2
TOKEN_VAR = 'CORREAD_REPORT_TOKEN'
TIMEOUT = 10
DEFAULT_EVERY = 60
STATUSES = ('in progress', 'complete', 'interrupted', 'discarded', 'failed', 'stopped')

# the fields of metadata.json that are sent; anything else stays local
VALIDATION_FIELDS = ('val_loss', 'val_loss_images', 'val_loss_words', 'val_accuracy_images', 'val_accuracy_words')
HISTORY_FIELDS = (
    (
        'epoch',
        'lr',
        'train_loss',
        'train_loss_images',
        'train_loss_words',
        'train_accuracy_images',
        'train_accuracy_words',
    )
    + VALIDATION_FIELDS
    + ('duration_seconds', 'finished_at')
)
INITIAL_VALIDATION_FIELDS = VALIDATION_FIELDS + ('duration_seconds', 'finished_at')
# what lesion runs add to their validations and epochs (see corread.lesion)
UNIT_FIELDS = (
    'mean_word_activation_lesioned',
    'mean_word_activation_other',
    'word_selective_lesioned',
    'word_selective_other',
)

# the share of an epoch each phase counts for in progress.fraction, as (start, width)
TRAINING_PHASES = {'training': (0.0, 0.95), 'validation': (0.95, 0.05)}
LESION_PHASES = {'training': (0.0, 0.95), 'validation': (0.95, 0.04), 'selectivity': (0.99, 0.01)}


def settings() -> tuple[str, str, float]:
    """``(url, token, every)`` from ``paths.toml`` and the environment.

    Called before a run starts, so a missing or unsafe setting fails at once rather than hours in.
    """
    try:
        section = get_section('report')
    except KeyError:
        raise ValueError('Reporting needs a [report] section with url = "https://..." in paths.toml') from None
    url = section.get('url')
    if not isinstance(url, str) or not url:
        raise ValueError('[report] in paths.toml needs url = "https://..."')
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'https' and not (parts.scheme == 'http' and _is_local(parts.hostname)):
        raise ValueError('[report] url must use https:// (plain http:// only for localhost), got: ' + url)
    token = os.environ.get(TOKEN_VAR)
    if not token:
        raise ValueError('Reporting needs the API token in the ' + TOKEN_VAR + ' environment variable')
    every = section.get('every', DEFAULT_EVERY)
    if not isinstance(every, (int, float)) or isinstance(every, bool) or every <= 0:
        raise ValueError('[report] every must be a positive number of seconds, got: ' + repr(every))
    return url, token, float(every)


def _is_local(host: str | None) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (TypeError, ValueError):
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # a redirect would resend the token to wherever it points, so none are followed
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'redirects are not followed (to ' + newurl + ')', headers, fp)


_opener = urllib.request.build_opener(_NoRedirect)


def post(url: str, token: str, payload: dict[str, Any]) -> None:
    """POST one report; raises on a network error or a non-2xx answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            'User-Agent': 'CORread-reporter/' + str(SCHEMA),
        },
    )
    try:
        with _opener.open(request, timeout=TIMEOUT) as response:
            response.read()
    except urllib.error.HTTPError as e:
        body = e.read(300).decode('utf-8', 'replace') if e.fp else ''
        raise RuntimeError(
            'report rejected by ' + url + ': HTTP ' + str(e.code) + ' ' + str(e.reason) + (' - ' + body if body else '')
        ) from None


# Per-run sequence numbers let the API drop reports that arrive out of order. They are milliseconds
# since 1970, so they keep increasing across resumes and processes.
_seq_lock = threading.Lock()
_last_seq = 0


def _next_seq() -> int:
    global _last_seq
    with _seq_lock:
        _last_seq = max(_last_seq + 1, time.time_ns() // 1_000_000)
        return _last_seq


class Reporter:
    """Sends reports from a background thread, so a slow or unreachable API never holds up the work.

    Only the newest unsent report is kept: a newer one replaces it, so nothing piles up while the API
    is down. Failures warn once, then stay quiet until a report gets through again.
    """

    def __init__(self, url: str, token: str, every: float):
        self.url, self.token, self.every = url, token, every
        self._cond = threading.Condition()
        self._pending: dict | None = None
        self._closed = False
        self._failing = False
        self._last_sent = 0.0
        self._thread = threading.Thread(target=self._run, name='corread-reporter', daemon=True)
        self._thread.start()

    def due(self) -> bool:
        """Whether a progress report is due (one every ``every`` seconds)."""
        return time.monotonic() - self._last_sent >= self.every

    def send(self, payload: dict[str, Any]) -> None:
        self._last_sent = time.monotonic()
        with self._cond:
            self._pending = payload
            self._cond.notify()

    def close(self, timeout: float = 15) -> None:
        """Send what is still pending, waiting at most ``timeout`` seconds."""
        with self._cond:
            self._closed = True
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._pending is None:
                    return
                payload, self._pending = self._pending, None
            try:
                post(self.url, self.token, payload)
                if self._failing:
                    logger.info('reporting to %s works again', self.url)
                self._failing = False
            except Exception as e:
                if not self._failing:
                    logger.warning(
                        'could not report status to %s (%s); the run continues and later reports will catch up',
                        self.url,
                        e,
                    )
                self._failing = True


def _error(e: BaseException | None) -> dict[str, str] | None:
    """Type and message of an exception, with the home folder shortened to ``~``."""
    if e is None:
        return None
    message = str(e).replace(os.path.expanduser('~'), '~')
    return {'type': type(e).__name__, 'message': message[:500]}


def _envelope(
    kind, meta, group, event, status, config, progress, eta_seconds, initial_validation, history, error
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError('unknown status ' + repr(status))
    return {
        'schema': SCHEMA,
        'seq': _next_seq(),
        'sent_at': now_iso(),
        'run_id': meta.get('run_id'),
        'kind': kind,
        'name': meta.get('name'),
        'group': group,
        'host': platform.node(),
        'event': event,
        'status': status,
        'started_at': meta.get('started_at'),
        'finished_at': meta.get('finished_at'),
        'config': config,
        'progress': progress,
        'eta_seconds': None if eta_seconds is None else max(0, round(eta_seconds)),
        'initial_validation': initial_validation,
        'history': history,
        'error': _error(error),
    }


def _fields(entry: dict | None, fields: tuple[str, ...]) -> dict | None:
    """The given fields of a validation or history entry, or ``None`` when it hasn't happened yet."""
    return {k: entry.get(k) for k in fields} if entry else None


def _model_config(meta: dict) -> dict[str, Any]:
    """The config fields training and lesion runs share: network, data, optimizer and machine."""
    args = meta.get('arguments') or {}
    model = meta.get('model') or {}
    env = meta.get('environment') or {}
    data = meta.get('data') or {}
    imageset, wordset = data.get('imageset') or {}, data.get('wordset') or {}
    return {
        'model_choice': args.get('model_choice') or model.get('model_choice'),
        'model_class': model.get('class'),
        'parameters': model.get('parameters'),
        'outputs': model.get('outputs'),
        'wordset': args.get('wordset') or (wordset.get('generation') or {}).get('name'),
        'batch_size': args.get('batch_size'),
        'lr': args.get('lr'),
        'momentum': args.get('momentum'),
        'weight_decay': args.get('weight_decay'),
        'img_classes': args.get('img_classes'),
        'wrd_classes': args.get('wrd_classes'),
        'train_images': (imageset.get('train_images') or 0) + (wordset.get('train_images') or 0) or None,
        'val_images': (imageset.get('val_images') or 0) + (wordset.get('val_images') or 0) or None,
        'device_name': env.get('device_name'),
        'gpus_used': env.get('gpus_used'),
    }


def _progress(meta: dict, history: list, status: str, live: tuple | None, phases: dict) -> tuple[dict, float | None]:
    """``(progress, eta_seconds)`` of a run; ``live`` is ``(phase, batch, batches)`` while it runs."""
    epochs = meta.get('epochs') or {}
    total, done = epochs.get('count'), len(history)
    phase = batch = batches = None
    epoch_fraction = 0.0
    if live is not None and status == 'in progress':
        phase, batch, batches = live
        if batches and phase in phases:
            start, width = phases[phase]
            epoch_fraction = start + width * min(batch / batches, 1.0)
    running = status == 'in progress' and total is not None and done < total
    fraction = None
    if total:
        fraction = 1.0 if status == 'complete' else min((done + epoch_fraction) / total, 1.0)
    eta = None
    durations = [h['duration_seconds'] for h in history if h.get('duration_seconds')]
    if running and durations:
        eta = (total - done - epoch_fraction) * sum(durations) / len(durations)
    return {
        'epochs_first': epochs.get('first'),
        'epochs_total': total,
        'epochs_completed': done,
        'epoch': (epochs.get('first') or 0) + done if running else None,
        'phase': phase,
        'batch': batch,
        'batches': batches,
        'fraction': fraction,
    }, eta


def training_payload(
    meta: dict, group: str, event: str, status: str, live: tuple | None = None, error: BaseException | None = None
) -> dict[str, Any]:
    """The report about a training run, from its ``metadata.json`` contents."""
    args = meta.get('arguments') or {}
    history = [_fields(h, HISTORY_FIELDS) for h in meta.get('history') or []]
    progress, eta = _progress(meta, history, status, live, TRAINING_PHASES)
    config = {'mode': meta.get('mode') or args.get('mode'), 'pre_name': args.get('pre_name'), **_model_config(meta)}
    return _envelope(
        'training',
        meta,
        group,
        event,
        status,
        config,
        progress,
        eta,
        _fields(meta.get('initial_validation'), INITIAL_VALIDATION_FIELDS),
        history,
        error,
    )


def lesion_payload(
    meta: dict, group: str, event: str, status: str, live: tuple | None = None, error: BaseException | None = None
) -> dict[str, Any]:
    """The report about a lesion run; its ``initial_validation`` holds the intact and the lesioned model's."""
    lesion = meta.get('lesion') or {}
    source = meta.get('initialised_from') or {}
    data = meta.get('data') or {}
    validation_fields = INITIAL_VALIDATION_FIELDS + UNIT_FIELDS
    history = [_fields(h, HISTORY_FIELDS + UNIT_FIELDS) for h in meta.get('history') or []]
    intact = _fields(meta.get('intact_validation'), validation_fields)
    lesioned = _fields(meta.get('initial_validation'), validation_fields)
    progress, eta = _progress(meta, history, status, live, LESION_PHASES)
    channel = lesion.get('word_channel')
    config = {
        'source_name': source.get('name'),
        'source_mode': source.get('mode'),
        **_model_config(meta),
        'units_total': lesion.get('units_total'),
        'lesioned_units': lesion.get('units'),
        'word_channel': {'first': channel[0], 'last': channel[1]} if channel else None,
        'reset_outgoing': lesion.get('reset_outgoing'),
        'silent_epochs': lesion.get('silent_epochs', 0),
        'selectivity_threshold': (lesion.get('selectivity') or {}).get('threshold'),
        'selectivity_images': (data.get('selectivity_images') or {}).get('images'),
        'selectivity_words': (data.get('wordset') or {}).get('val_images'),
    }
    initial = {'intact': intact, 'lesioned': lesioned} if intact or lesioned else None
    return _envelope('lesion', meta, group, event, status, config, progress, eta, initial, history, error)


def wordset_payload(
    meta: dict, event: str, status: str, words_done: int | None = None, error: BaseException | None = None
) -> dict[str, Any]:
    """The report about a word set; ``words_done`` is ``None`` when not known (outside the generating process)."""
    args = meta.get('arguments') or {}
    total = len(meta.get('words') or []) or args.get('num_words')
    if status == 'complete':
        words_done = total
    fraction = words_done / total if words_done is not None and total else None
    eta = None
    started = meta.get('started_at')
    if status == 'in progress' and words_done and total and started:
        elapsed = time.time() - datetime.fromisoformat(started).timestamp()
        eta = elapsed / words_done * (total - words_done)
    config = {
        k: args.get(k)
        for k in ('wordlist', 'num_words', 'num_train', 'num_val', 'train_fonts', 'val_fonts', 'width', 'height')
    }
    return _envelope(
        'wordset',
        meta,
        'wordsets',
        event,
        status,
        config,
        {'words_done': words_done, 'words_total': total, 'fraction': fraction},
        eta,
        None,
        None,
        error,
    )
