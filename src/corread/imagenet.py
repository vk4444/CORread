# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Turn the ImageNet (ILSVRC 2012) archives into the ``train/<wnid>/`` and ``val/<wnid>/`` folders training reads.

ImageNet is not distributed with CORread: download ``ILSVRC2012_img_train.tar``,
``ILSVRC2012_img_val.tar`` and ``ILSVRC2012_devkit_t12.tar.gz`` from https://image-net.org under its
terms of access, then run :func:`prepare_imagenet`.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tarfile
from collections.abc import Iterator

from tqdm import tqdm

from ._io import ordered_map, rename_dir
from .paths import get_path

logger = logging.getLogger(__name__)

TRAIN_ARCHIVE = 'ILSVRC2012_img_train.tar'
VAL_ARCHIVE = 'ILSVRC2012_img_val.tar'
DEVKIT_ARCHIVE = 'ILSVRC2012_devkit_t12.tar.gz'


def _write_file(item: tuple[str, bytes]) -> None:
    path, data = item
    with open(path, 'wb') as f:
        f.write(data)


def _plain_name(name: str, archive_path: str) -> str:
    """An archive entry's name, refused unless it is a plain file name (no folders, no ``..``)."""
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Unexpected entry in ' + archive_path + ': ' + name)
    return name


def _val_labels(devkit_path: str) -> dict[str, str]:
    """``{val image file name: wnid}`` from the devkit's ``meta.mat`` and ground truth list."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("Reading the ImageNet devkit needs scipy: pip install 'corread[imagenet]'") from None
    with tarfile.open(devkit_path, 'r:gz') as tar:
        meta = tar.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()
        truth = tar.extractfile('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt').read()
    # the 1000 leaf synsets are the classes; the rest are inner WordNet nodes
    synsets = loadmat(io.BytesIO(meta), squeeze_me=True)['synsets']
    wnids = {int(s['ILSVRC2012_ID']): str(s['WNID']) for s in synsets if int(s['num_children']) == 0}
    ids = [int(line) for line in truth.decode().split()]
    if len(wnids) != 1000 or len(ids) != 50000:
        raise ValueError('Unexpected devkit contents in ' + devkit_path)
    return {'ILSVRC2012_val_%08d.JPEG' % (i + 1): wnids[c] for i, c in enumerate(ids)}


def _unpack_val(archive_path: str, devkit_path: str, dest: str, workers: int, keep_archive: bool) -> None:
    """Sort the flat val archive into ``dest/<wnid>/``, through ``val.part/`` renamed when complete."""
    labels = _val_labels(devkit_path)
    part = dest + '.part'
    shutil.rmtree(part, ignore_errors=True)
    for wnid in set(labels.values()):
        os.makedirs(os.path.join(part, wnid), exist_ok=True)

    def members() -> Iterator[tuple[str, bytes]]:
        with tarfile.open(archive_path, 'r:') as tar:
            for m in tar:
                name = _plain_name(m.name, archive_path)
                if not m.isfile() or name not in labels:
                    raise ValueError('Unexpected entry in ' + archive_path + ': ' + m.name)
                yield os.path.join(part, labels[name], name), tar.extractfile(m).read()

    count = sum(
        1
        for _ in tqdm(
            ordered_map(_write_file, members(), workers, workers * 8),
            total=len(labels),
            desc='Unpacking val',
            unit='file',
        )
    )
    if count != len(labels):
        raise RuntimeError(archive_path + ' has ' + str(count) + ' images, expected ' + str(len(labels)))
    rename_dir(part, dest)
    if not keep_archive:
        os.remove(archive_path)


def _unpack_train(archive_path: str, dest: str, workers: int, keep_archive: bool) -> None:
    """Unpack the train archive (one ``<wnid>.tar`` per class) into ``dest/<wnid>/``.

    Unless ``keep_archive``, it works from the last class to the first and truncates the archive after
    each, so the disk never holds much more than one copy of the data. Each class goes into
    ``<wnid>.part/`` and is renamed when complete, so an interrupted run picks up where it stopped.
    """
    os.makedirs(dest, exist_ok=True)
    with open(archive_path, 'rb' if keep_archive else 'r+b') as f:
        # a fully consumed archive is an empty file, which tarfile refuses
        f.seek(0, os.SEEK_END)
        members = []
        if f.tell() > 0:
            f.seek(0)
            # closing the TarFile leaves f open, as tarfile only closes files it opened itself
            with tarfile.open(fileobj=f, mode='r:') as tar:
                members = tar.getmembers()
        for m in members:
            name = _plain_name(m.name, archive_path)
            if not m.isfile() or not name.endswith('.tar'):
                raise ValueError('Unexpected entry in ' + archive_path + ': ' + m.name)

        for m in tqdm(list(reversed(members)), desc='Unpacking train', unit='class'):
            folder = os.path.join(dest, m.name[: -len('.tar')])
            if not os.path.isdir(folder):
                part = folder + '.part'
                shutil.rmtree(part, ignore_errors=True)
                os.makedirs(part, exist_ok=True)
                f.seek(m.offset_data)
                data = f.read(m.size)

                def images(data=data, part=part, member=m) -> Iterator[tuple[str, bytes]]:
                    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as inner:
                        for im in inner:
                            if not im.isfile():
                                raise ValueError('Unexpected entry in ' + member.name + ': ' + im.name)
                            yield os.path.join(part, _plain_name(im.name, member.name)), inner.extractfile(im).read()

                for _ in ordered_map(_write_file, images(), workers, workers * 8):
                    pass
                rename_dir(part, folder)
            if not keep_archive:
                f.truncate(m.offset)
    if not keep_archive:
        os.remove(archive_path)


def prepare_imagenet(folder: str | None = None, keep_archives: bool = False, workers: int = 16) -> None:
    """Unpack the ImageNet archives in ``folder`` (by default ``imagesets`` from ``paths.toml``).

    The archives are used up as they are unpacked (``keep_archives=True`` keeps them, but then the disk
    needs room for both). Entries are written only as plain files inside their class folders, never
    where an archive's paths point. Safe to rerun after an interruption: it continues where it stopped.
    """
    folder = folder if folder is not None else get_path('imagesets')
    train_tar, val_tar = os.path.join(folder, TRAIN_ARCHIVE), os.path.join(folder, VAL_ARCHIVE)
    devkit = os.path.join(folder, DEVKIT_ARCHIVE)
    train_dir, val_dir = os.path.join(folder, 'train'), os.path.join(folder, 'val')

    # an existing val/ is complete, since it is renamed into place last
    val_todo = not os.path.isdir(val_dir)
    train_todo = os.path.isfile(train_tar)
    missing = [
        p
        for p, needed in ((val_tar, val_todo), (devkit, val_todo), (train_tar, not os.path.isdir(train_dir)))
        if needed and not os.path.isfile(p)
    ]
    if missing:
        raise FileNotFoundError('Missing: ' + ', '.join(missing))

    # val first: it is quick and checks the devkit before the long train step
    if val_todo:
        _unpack_val(val_tar, devkit, val_dir, workers, keep_archives)
    elif os.path.isfile(val_tar) and not keep_archives:
        os.remove(val_tar)
    if train_todo:
        _unpack_train(train_tar, train_dir, workers, keep_archives)

    # ImageFolder numbers classes by sorted folder name, so train and val need the same folders
    train_classes, val_classes = sorted(os.listdir(train_dir)), sorted(os.listdir(val_dir))
    if train_classes != val_classes or len(train_classes) != 1000:
        raise RuntimeError('train/ and val/ do not have the same 1000 class folders')
    logger.info('ImageNet ready in %s', folder)
