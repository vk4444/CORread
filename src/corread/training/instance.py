# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2021 T. Hannagan, A. Agrawal, L. Cohen, S. Dehaene (Origins-of-VWFA)
# Copyright (C) 2026 Vojtěch Kubala
#
# The training procedure (pre-literate training on images, then literate training on images and
# words together) follows https://github.com/THANNAGA/Origins-of-VWFA (clean_train.py), via
# Dr. Olga Solaja's adaptation. Rewritten by Vojtěch Kubala, 2026: resumable runs with full
# provenance, validation of the starting weights, CORnet-S, multi-GPU training, status reports
# and Hugging Face Hub export.
"""One training run: a model trained in one mode, in its own folder, resumable and fully recorded."""

from __future__ import annotations

import gc
import glob
import hashlib
import logging
import os
import platform
import secrets
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
from tqdm import tqdm

from .. import reporting
from .._io import check_name, now_iso, read_json, write_json
from .._provenance import code_provenance, sha256_file
from ..hub._client import read_sidecar
from ..hub._models import discard_model_export, publish_model, start_model_export
from ..models import CORnet, build_network
from ..paths import get_path
from . import descriptions
from .data import image_dataset, word_dataset
from .metrics import accuracy, batch_sums, duration_text, means

logger = logging.getLogger(__name__)

MODES = ('pre', 'illit', 'lit_bias', 'lit_no_bias')
LITERATE_MODES = ('lit_bias', 'lit_no_bias')
"""The modes that train on images and words together, with the word outputs after the image outputs."""
PRE_FOLDER = 'pre'
ILLIT_FOLDER = 'illiterate'
MODEL_CHOICES = ('z', 's')

# arguments that may change when resuming, since they don't change the result
RESUME_FREE_ARGS = ('num_workers', 'gpus', 'resume', 'restore_path')


def _unwrap(net: nn.Module) -> nn.Module:
    """The network inside an ``nn.DataParallel``, or the network itself."""
    return net.module if hasattr(net, 'module') else net


def _host_id() -> str:
    # identifies this machine in metadata.json without recording its name
    return hashlib.sha256(platform.node().encode('utf-8')).hexdigest()[:16]


def _still_running(process: dict | None) -> bool:
    """Whether the process recorded in a model's metadata is alive on this machine."""
    if not process or process.get('host_id') != _host_id() or process.get('pid') == os.getpid():
        return False
    try:
        os.kill(process['pid'], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class _Setup:
    """What a run trains: the network, the data and the score arrays it continues."""

    net: nn.Module
    training: data.DataLoader
    validation: data.DataLoader
    data_info: dict
    pretrained: dict | None
    cat_scores: np.ndarray
    trainloss: list
    valloss: list
    max_epochs: int
    shift_epoch: int


class TrainingInstance:
    """Train one model in one mode.

    The modes follow Hannagan et al. (2021): a CORnet is first trained on ImageNet (``pre``), then
    either keeps training on images alone (``illit``) or learns to read: it trains on the images and
    the words of a word set together, with the word classes added after the image classes. In
    ``lit_bias`` the word outputs read only the last 49 dense-layer units; in ``lit_no_bias`` they read
    all 512.

    Each model gets its own folder in the ``models`` folder of ``paths.toml``:

    - ``pre``: ``pre/<name or pre_z_49>/``
    - ``illit``: ``illiterate/<name or illit_z_79>/``
    - ``lit_bias``, ``lit_no_bias``: ``<name or word set>/lit_bias_z_79/``

    where ``z`` is the model choice and 49 or 79 the last epoch, counted from 0 over the pre-literate
    and then the later epochs. The folder holds the finished model ``<folder name>.pth.tar``, a
    checkpoint after every epoch, the scores and losses as ``.npy`` arrays, and ``metadata.json``:
    how the model was trained, on which data (with hashes), by which code, and the history of every
    epoch.

    Parameters
    ----------
    model_choice : {'z', 's'}
        CORnet-Z or CORnet-S.
    mode : {'pre', 'illit', 'lit_bias', 'lit_no_bias'}
        What to train; the literate and illiterate modes start from a pre-literate model.
    name : str, optional
        Folder name of a ``pre`` or ``illit`` model; for the literate modes the name of their group
        folder (by default the word set's name).
    pre_name : str, optional
        The pre-literate model to start from; by default ``pre_<model_choice>_<max_epochs_pre - 1>``.
    wordset : str, optional
        The word set (literate modes), by its name in the ``wordsets`` folder.
    wrd_path, img_path, models_dir : str, optional
        Folders of the word set, the images (ImageNet-style ``train/`` and ``val/``) and the models;
        by default from ``paths.toml``.
    restore_path : str, optional
        A checkpoint to continue from.
    resume : bool
        Continue an interrupted run from its last checkpoint.
    export : bool
        Upload the run to the Hugging Face Hub as it trains (see :mod:`corread.hub`).
    report : bool
        Send the run's status to the reporting API set in ``paths.toml`` (see :mod:`corread.reporting`).
    seed : int, optional
        Seed of the run; a random one is drawn and recorded when not given.
    num_workers : int
        DataLoader worker processes.
    gpus : int or list of int, optional
        GPUs to train on: all (``None``), the first ``n``, or these indices; ``0`` trains on the CPU.
    batch_size : int
        Training batch size.
    num_val_items : int
        Validation batch size; with as many val images per class, each batch is one class.
    max_epochs_pre, max_epochs_lit : int
        Epochs of pre-literate training, and of the literate or illiterate training after it.
    img_classes, wrd_classes : int
        Number of image and word classes.
    lr, momentum, weight_decay : float
        SGD settings; there is no learning-rate schedule.
    """

    def __init__(
        self,
        model_choice: str = 'z',
        mode: str = 'pre',
        name: str | None = None,
        pre_name: str | None = None,
        *,
        wordset: str | None = None,
        wrd_path: str | None = None,
        img_path: str | None = None,
        models_dir: str | None = None,
        restore_path: str | None = None,
        resume: bool = False,
        export: bool = False,
        report: bool = False,
        seed: int | None = None,
        num_workers: int = 1,
        gpus: int | list[int] | None = None,
        batch_size: int = 100,
        num_val_items: int = 50,
        max_epochs_pre: int = 50,
        max_epochs_lit: int = 30,
        img_classes: int = 1000,
        wrd_classes: int = 1000,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        if mode not in MODES:
            raise ValueError('mode must be one of ' + ', '.join(MODES) + ', got: ' + repr(mode))
        if model_choice not in MODEL_CHOICES:
            raise ValueError("model_choice must be 'z' (CORnet-Z) or 's' (CORnet-S), got: " + repr(model_choice))
        for value, what in ((name, 'name'), (pre_name, 'pre_name'), (wordset, 'wordset')):
            if value is not None:
                check_name(value, what)
        if wordset is not None and wrd_path is not None:
            raise ValueError('Pass either wordset or wrd_path, not both.')
        self.model_choice = model_choice
        self.mode = mode
        self.name = name
        self.pre_name = pre_name if pre_name is not None else 'pre_' + model_choice + '_' + str(max_epochs_pre - 1)
        self.wordset = wordset
        self.wrd_path = os.path.join(get_path('wordsets'), wordset) if wordset is not None else wrd_path
        self.restore_path = restore_path
        self.batch_size = batch_size
        self.num_val_items = num_val_items
        self.max_epochs_pre = max_epochs_pre
        self.max_epochs_lit = max_epochs_lit
        self.img_classes = img_classes
        self.wrd_classes = wrd_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._init_run(
            resume=resume,
            export=export,
            report=report,
            seed=seed,
            models_dir=models_dir,
            img_path=img_path,
            num_workers=num_workers,
            gpus=gpus,
        )

    def _init_run(
        self,
        *,
        resume: bool,
        export: bool,
        report: bool,
        seed: int | None,
        models_dir: str | None,
        img_path: str | None,
        num_workers: int,
        gpus: int | list[int] | None,
    ) -> None:
        """The settings every run has, checked now rather than hours into the run."""
        self.resume = resume
        self.export = export
        self.report = report
        if report:
            reporting.settings()
        self.seed = seed
        self.models_dir = models_dir if models_dir is not None else get_path('models')
        self.img_path = img_path if img_path is not None else get_path('imagesets')
        self.num_workers = num_workers
        self.gpus = gpus
        self._gpu_ids()
        self.model_dir, self.model_name = self._output_location()
        # while training: the reporter, the run's metadata and where in the epoch it is
        self._reporter: reporting.Reporter | None = None
        self._live_meta: dict | None = None
        self._live: tuple | None = None
        # while training with export=True: (Hub repo, run_id)
        self._export_to: tuple[str, str] | None = None

    # --- where the model goes, and what it was trained with -------------------------------------

    def _output_location(self) -> tuple[str, str]:
        """``(folder the model goes into, name of the model and that folder)``."""
        last_epoch = self.max_epochs_pre - 1 if self.mode == 'pre' else self.max_epochs_pre + self.max_epochs_lit - 1
        auto = self.mode + '_' + self.model_choice + '_' + str(last_epoch)
        if self.mode == 'pre':
            parent, model_name = os.path.join(self.models_dir, PRE_FOLDER), self.name or auto
        elif self.mode == 'illit':
            parent, model_name = os.path.join(self.models_dir, ILLIT_FOLDER), self.name or auto
        else:
            # literate models are grouped by name, by default the word set's
            group = (
                self.name
                or self.wordset
                or (os.path.basename(os.path.normpath(self.wrd_path)) if self.wrd_path else None)
            )
            if group is None:
                raise ValueError("Literate training needs a word set: pass wordset='<name>' or wrd_path.")
            if group in (PRE_FOLDER, ILLIT_FOLDER):
                raise ValueError('name ' + repr(group) + ' is reserved for the ' + group + ' models folder.')
            parent, model_name = os.path.join(self.models_dir, group), auto
        return os.path.join(parent, model_name), model_name

    def _group(self) -> str:
        """The folder this model's folder is in: ``pre``, ``illiterate`` or the word set's."""
        return os.path.basename(os.path.dirname(self.model_dir))

    def _arguments(self) -> dict[str, Any]:
        """The arguments of this run, as recorded in ``metadata.json``."""
        return {
            'name': self.name,
            'mode': self.mode,
            'model_choice': self.model_choice,
            'pre_name': self.pre_name if self.mode != 'pre' else None,
            'wordset': self.wordset,
            'wrd_path': os.path.abspath(self.wrd_path) if self.wrd_path else None,
            'img_path': os.path.abspath(self.img_path),
            'models_dir': os.path.abspath(self.models_dir),
            'restore_path': os.path.abspath(self.restore_path) if self.restore_path else None,
            'resume': self.resume,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'num_val_items': self.num_val_items,
            'max_epochs_pre': self.max_epochs_pre,
            'max_epochs_lit': self.max_epochs_lit,
            'img_classes': self.img_classes,
            'wrd_classes': self.wrd_classes,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
        }

    def _changed_arguments(self, old: dict) -> list[str]:
        """How this run's arguments differ from recorded ones, leaving out those that don't change the model."""
        new = self._arguments()
        ignored = RESUME_FREE_ARGS + (('wordset', 'wrd_path') if self.mode in ('pre', 'illit') else ())
        return [
            k + ' was ' + repr(old.get(k)) + ', now ' + repr(new[k])
            for k in new
            if k not in ignored and old.get(k) != new[k]
        ]

    def is_complete(self) -> bool:
        """Whether this model is already trained; raises if it was trained with different arguments."""
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if not os.path.isfile(meta_path):
            return False
        meta = read_json(meta_path)
        if meta.get('status') != 'complete':
            return False
        changed = self._changed_arguments(meta['arguments'])
        if changed:
            raise ValueError(
                self.model_dir
                + ' was trained with different arguments: '
                + ', '.join(changed)
                + ' (delete it or choose another name)'
            )
        return True

    # --- the folder of an earlier run -----------------------------------------------------------

    def _nothing_to_resume(self) -> bool:
        """Whether the model folder holds only what a run writes before its first epoch finishes."""
        for root, _, files in os.walk(self.model_dir):
            for f in files:
                path = os.path.relpath(os.path.join(root, f), self.model_dir)
                if path not in ('metadata.json', 'metadata.json.tmp', 'training_eta_log.txt') and not f.endswith(
                    '.part'
                ):
                    return False
        return True

    def _check_output(self) -> dict | None:
        """Check the model folder before anything loads; the metadata of a run to continue, else ``None``."""
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if not os.path.exists(self.model_dir):
            return None
        if not os.path.isfile(meta_path):
            raise FileExistsError('Already exists: ' + self.model_dir + ' (delete it or choose another name)')
        meta = read_json(meta_path)
        if meta.get('status') == 'complete':
            raise FileExistsError('Model already trained: ' + self.model_dir + ' (delete it or choose another name)')
        if _still_running(meta.get('process')):
            raise RuntimeError(
                'A run is still training in ' + self.model_dir + ' (process ' + str(meta['process']['pid']) + ')'
            )
        if self._nothing_to_resume():
            # left by a run stopped before its first epoch finished
            shutil.rmtree(self.model_dir)
            return None
        if not self.resume:
            raise FileExistsError(
                'An unfinished run is in '
                + self.model_dir
                + ' ('
                + str(meta.get('epochs_completed'))
                + ' epochs done); pass resume=True to continue it, or delete the folder'
            )
        changed = self._changed_arguments(meta['arguments'])
        if changed:
            raise ValueError('Cannot resume ' + self.model_dir + ' with different arguments: ' + ', '.join(changed))
        return meta

    def _record_interruption(self) -> None:
        """After Ctrl+C: remove the model folder if it holds nothing to resume from, else mark it interrupted."""
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if not os.path.isfile(meta_path):
            return
        meta = read_json(meta_path)
        if meta.get('status') == 'complete' or meta.get('process', {}).get('pid') != os.getpid():
            return
        if self._nothing_to_resume():
            shutil.rmtree(self.model_dir)
            logger.warning('Interrupted before the first epoch finished; removed %s', self.model_dir)
            return
        meta['status'] = 'interrupted'
        meta.setdefault('interrupted_at', []).append(now_iso())
        write_json(meta_path, meta)
        logger.warning(
            'Interrupted after %s epoch(s) in %s; pass resume=True to continue from the last checkpoint',
            meta.get('epochs_completed'),
            self.model_dir,
        )

    # --- devices and data -----------------------------------------------------------------------

    def _gpu_ids(self) -> list[int]:
        """The CUDA device indices to train on, from ``gpus``."""
        available = torch.cuda.device_count()
        if self.gpus is None:
            return list(range(available))
        if isinstance(self.gpus, int) and not isinstance(self.gpus, bool):
            if not 0 <= self.gpus <= available:
                raise ValueError('gpus=' + str(self.gpus) + ' but ' + str(available) + ' GPU(s) are available')
            return list(range(self.gpus))
        ids = list(self.gpus) if isinstance(self.gpus, (list, tuple)) else None
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 0 <= i < available for i in ids):
            raise ValueError(
                'gpus must be None, a number or a list of distinct GPU indices below '
                + str(available)
                + ', got: '
                + repr(self.gpus)
            )
        return ids

    def _device(self, gpu_ids: list[int]) -> torch.device:
        if gpu_ids:
            device = torch.device('cuda', gpu_ids[0])
            torch.cuda.set_device(device)
        elif self.gpus is None and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        return device

    def _loader(self, dataset, batch_size: int, shuffle: bool, device: torch.device) -> data.DataLoader:
        # persistent workers and prefetching keep the GPU fed; they need worker processes
        extra = dict(persistent_workers=True, prefetch_factor=4) if self.num_workers > 0 else {}
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=device.type == 'cuda',
            **extra,
        )

    def _image_loaders(self, device: torch.device) -> tuple[data.DataLoader, data.DataLoader, dict]:
        """The image-only loaders of ``pre`` and ``illit``, and their data info."""
        train_set = image_dataset(self.img_path, 'train')
        val_set = image_dataset(self.img_path, 'val')
        training = self._loader(train_set, self.batch_size, True, device)
        validation = self._loader(val_set, self.num_val_items, False, device)
        return (
            training,
            validation,
            {'imageset': self._dataset_info(self.img_path, train_set, val_set), 'wordset': None},
        )

    def _literate_loaders(self, device: torch.device) -> tuple[data.DataLoader, data.DataLoader, dict]:
        """The images and the words together, the words labelled after the image classes."""
        train_words = word_dataset(self.wrd_path, 'train', label_offset=self.img_classes)
        train_images = image_dataset(self.img_path, 'train')
        val_words = word_dataset(self.wrd_path, 'val', label_offset=self.img_classes)
        val_images = image_dataset(self.img_path, 'val')
        training = self._loader(data.ConcatDataset((train_images, train_words)), self.batch_size, True, device)
        validation = self._loader(data.ConcatDataset((val_images, val_words)), self.num_val_items, False, device)
        data_info = {
            'imageset': self._dataset_info(self.img_path, train_images, val_images),
            'wordset': self._dataset_info(self.wrd_path, train_words, val_words),
        }
        data_info['wordset']['label_offset'] = self.img_classes
        return training, validation, data_info

    def _dataset_info(self, path: str, train_set, val_set) -> dict[str, Any]:
        """Size, classes and transforms of a dataset; for a generated word set also how it was made."""
        info: dict[str, Any] = {
            'path': os.path.abspath(path),
            'train_images': len(train_set),
            'val_images': len(val_set),
            'classes': len(train_set.classes),
            'first_class': train_set.classes[0],
            'last_class': train_set.classes[-1],
            'train_and_val_classes_match': train_set.classes == val_set.classes,
            'class_order': 'sorted folder names (torchvision ImageFolder)',
            'train_transform': str(train_set.transform),
            'val_transform': str(val_set.transform),
        }
        meta_path = os.path.join(path, 'metadata.json')
        if os.path.isfile(meta_path):
            generated = read_json(meta_path)
            if generated.get('status') != 'complete':
                raise RuntimeError(path + ' is not complete (status: ' + str(generated.get('status')) + ')')
            info['metadata_sha256'] = sha256_file(meta_path)
            info['generation'] = {k: generated.get(k) for k in ('name', 'arguments', 'seed_used', 'generator')}
            checksums_path = os.path.join(path, 'checksums.json')
            if os.path.isfile(checksums_path):
                info['images_digest'] = read_json(checksums_path).get('total')
            regeneration_path = os.path.join(path, 'regeneration.json')
            if os.path.isfile(regeneration_path):
                regeneration = read_json(regeneration_path)
                info['regenerated'] = {k: regeneration.get(k) for k in ('regenerated_at', 'verified', 'pillow')}
            hub = read_sidecar(path)
            if hub is not None:
                info['hub'] = hub
        return info

    # --- the network and data of the run --------------------------------------------------------

    def _pre_files(self) -> tuple[str, str]:
        """The folder and model file of the pre-literate model a literate or illiterate run starts from."""
        pre_dir = os.path.join(self.models_dir, PRE_FOLDER, self.pre_name)
        pre_file = os.path.join(pre_dir, self.pre_name + '.pth.tar')
        if not os.path.isfile(pre_file):
            raise FileNotFoundError('Pre-literate model not found: ' + pre_file + ' (train it first or pass pre_name)')
        return pre_dir, pre_file

    def _load_pre(self) -> tuple[CORnet, dict]:
        """The pre-literate network a literate or illiterate run starts from, and its record for ``metadata.json``."""
        pre_dir, pre_file = self._pre_files()
        net_pre = build_network(self.model_choice, 'pre', img_classes=self.img_classes)
        ckpt = torch.load(pre_file, map_location='cpu', weights_only=True)
        net_pre.load_state_dict(ckpt['state_dict'])
        if ckpt['epoch'] != self.max_epochs_pre - 1:
            raise ValueError(
                'Pre-literate model '
                + self.pre_name
                + ' ended at epoch '
                + str(ckpt['epoch'])
                + ', but max_epochs_pre='
                + str(self.max_epochs_pre)
                + ' expects '
                + str(self.max_epochs_pre - 1)
            )
        return net_pre, self._pretrained_info(pre_dir, pre_file, ckpt)

    def _pretrained_info(self, pre_dir: str, pre_file: str, ckpt: dict) -> dict[str, Any]:
        """Which pre-literate model a literate or illiterate model starts from, for ``metadata.json``."""
        info: dict[str, Any] = {
            'name': self.pre_name,
            'path': os.path.abspath(pre_dir),
            'file_sha256': sha256_file(pre_file),
            'epoch': ckpt['epoch'],
        }
        meta_path = os.path.join(pre_dir, 'metadata.json')
        if os.path.isfile(meta_path):
            pre_meta = read_json(meta_path)
            info['metadata_sha256'] = sha256_file(meta_path)
            info['seed_used'] = pre_meta.get('seed_used')
            info['arguments'] = pre_meta.get('arguments')
        hub = read_sidecar(pre_dir)
        if hub is not None:
            info['hub'] = hub
        return info

    def _build(self, device: torch.device) -> _Setup:
        """The network, data and score arrays of the run."""
        if self.mode == 'pre':
            training, validation, data_info = self._image_loaders(device)
            net = build_network(self.model_choice, 'pre', img_classes=self.img_classes)
            return _Setup(
                net,
                training,
                validation,
                data_info,
                None,
                np.zeros((self.max_epochs_pre, self.img_classes)),
                [],
                [],
                self.max_epochs_pre,
                0,
            )

        pre_dir, _ = self._pre_files()
        if self.mode == 'illit':
            # the pre-literate model keeps training on the images alone
            training, validation, data_info = self._image_loaders(device)
            classes = self.img_classes
        else:
            if self.wrd_path is None:
                raise ValueError("Literate training needs a word set: pass wordset='<name>' or wrd_path.")
            training, validation, data_info = self._literate_loaders(device)
            classes = self.img_classes + self.wrd_classes
        # the scores and losses continue the pre-literate model's
        cat_scores = np.zeros((self.max_epochs_pre + self.max_epochs_lit, classes))
        cat_scores[: self.max_epochs_pre, : self.img_classes] = np.load(os.path.join(pre_dir, 'cat_scores.npy'))[
            : self.max_epochs_pre
        ]
        trainloss = np.load(os.path.join(pre_dir, 'trainloss.npy')).tolist()
        valloss = np.load(os.path.join(pre_dir, 'valloss.npy')).tolist()
        net, pretrained = self._load_pre()
        if self.mode in LITERATE_MODES:
            net = build_network(self.model_choice, self.mode, self.img_classes, self.wrd_classes, pretrained=net)
        logger.info('built the %s network of %s', self.mode, self.model_name)
        return _Setup(
            net,
            training,
            validation,
            data_info,
            pretrained,
            cat_scores,
            trainloss,
            valloss,
            self.max_epochs_lit,
            self.max_epochs_pre,
        )

    # --- metadata -------------------------------------------------------------------------------

    def _environment(self, device: torch.device, gpu_ids: list[int]) -> dict[str, Any]:
        cuda = device.type == 'cuda'
        return {
            **code_provenance(),
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'numpy': np.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version() if cuda else None,
            'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if cuda else platform.processor(),
            'gpus_used': len(gpu_ids),
            'gpu_ids': gpu_ids,
            'platform': platform.platform(),
        }

    def _metadata(
        self,
        run_id: str,
        start_time: float,
        seed_used: int,
        net: nn.Module,
        optimizer,
        setup: _Setup,
        device: torch.device,
        gpu_ids: list[int],
        restore_path: str | None,
    ) -> dict[str, Any]:
        model = _unwrap(net)
        literate = self.mode in LITERATE_MODES
        pretrained = setup.pretrained
        if pretrained is None and restore_path is not None:
            pretrained = {'checkpoint': os.path.abspath(restore_path), 'file_sha256': sha256_file(restore_path)}
        return {
            'name': self.model_name,
            'run_id': run_id,
            'mode': self.mode,
            'status': 'in progress',
            'started_at': now_iso(start_time),
            'finished_at': None,
            'training_seconds': None,
            'epochs_completed': 0,
            'epochs': {
                'first': setup.shift_epoch,
                'last': setup.shift_epoch + setup.max_epochs - 1,
                'count': setup.max_epochs,
                'numbering': descriptions.EPOCH_NUMBERING,
            },
            'arguments': self._arguments(),
            'seed_used': seed_used,
            'reproduce': descriptions.REPRODUCE,
            'initialised_from': pretrained,
            'model': {
                'class': type(model).__name__,
                'model_choice': self.model_choice,
                'parameters': sum(p.numel() for p in model.parameters()),
                'outputs': self.img_classes + (self.wrd_classes if literate else 0),
                'output_layout': 'image classes 0..img_classes-1' + (', then word classes' if literate else ''),
            },
            'data': dict(
                setup.data_info,
                batch_size=self.batch_size,
                val_batch_size=self.num_val_items,
                train_shuffle=True,
                val_shuffle=False,
            ),
            'optimization': {
                'loss': 'CrossEntropyLoss',
                'optimizer': {
                    'type': type(optimizer).__name__,
                    'lr': self.lr,
                    'momentum': self.momentum,
                    'weight_decay': self.weight_decay,
                },
                'scheduler': descriptions.SCHEDULER,
                'lr_used': descriptions.LR_USED,
            },
            'epoch_procedure': descriptions.EPOCH_PROCEDURE,
            'metrics': descriptions.METRICS,
            'cat_scores': descriptions.cat_scores(literate),
            'environment': self._environment(device, gpu_ids),
            'files': descriptions.files(self.model_name, continues_pre=self.mode != 'pre'),
            'initial_validation': None,
            'history': [],
        }

    # --- training -------------------------------------------------------------------------------

    def train(self) -> None:
        """Train the model.

        Ctrl+C stops cleanly: the DataLoader workers and GPUs are released, and the model folder is
        removed if no epoch finished, or kept with status ``interrupted`` for ``resume=True``.
        """
        self._reporter = reporting.Reporter(*reporting.settings()) if self.report else None
        try:
            self._train()
        except KeyboardInterrupt as e:
            # Ctrl+C often lands while DataParallel waits for its per-GPU threads; if Python exits while
            # they still run, the process aborts inside PyTorch, so let them finish first
            for t in threading.enumerate():
                if t is not threading.current_thread() and not t.daemon:
                    t.join(timeout=30)
            # the interrupted frames hold the DataLoaders; clearing them shuts their workers down now
            traceback.clear_frames(e.__traceback__)
            gc.collect()
            for i in self._gpu_ids():
                torch.cuda.synchronize(i)
            self._record_interruption()
            self._send_report('interrupted', 'interrupted' if os.path.isdir(self.model_dir) else 'discarded')
            self._stop_export('interrupted')
            raise
        except Exception as e:
            self._send_report('failed', 'failed', error=e)
            self._stop_export('failed')
            raise
        finally:
            if self._reporter is not None:
                self._reporter.close()
            self._reporter, self._live_meta, self._live = None, None, None
            self._export_to = None

    def _train(self) -> None:
        start_time = time.time()
        # fail before the slow dataset loading if the output is taken, here or (export=True) on the Hub
        resumed_meta = self._check_output()
        run_id = resumed_meta['run_id'] if resumed_meta is not None else str(uuid.uuid4())
        if self.export:
            self._export_to = (start_model_export(self._group(), self.model_name, run_id), run_id)
        checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        restore_path = self.restore_path
        if resumed_meta is not None and restore_path is None:
            found = sorted(glob.glob(os.path.join(checkpoints_dir, 'epoch_*.pth.tar')))
            restore_path = found[-1] if found else None

        # a restored checkpoint brings back its own random state
        seed_used = (
            resumed_meta['seed_used']
            if resumed_meta is not None
            else (self.seed if self.seed is not None else secrets.randbits(63))
        )
        torch.manual_seed(seed_used)
        gpu_ids = self._gpu_ids()
        device = self._device(gpu_ids)
        logger.info('training %s on %s', self.model_dir, device)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        setup = self._build(device)
        net = setup.net
        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=gpu_ids)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

        start_epoch, history = 0, []
        if restore_path is not None:
            start_epoch, history = self._restore(restore_path, net, optimizer, setup)

        os.makedirs(checkpoints_dir, exist_ok=True)
        meta_path = os.path.join(self.model_dir, 'metadata.json')
        if resumed_meta is not None:
            meta = resumed_meta
            meta['status'] = 'in progress'
            meta.setdefault('resumed', []).append(
                {
                    'at': now_iso(start_time),
                    'from_checkpoint': os.path.abspath(restore_path) if restore_path else None,
                    'environment': self._environment(device, gpu_ids),
                }
            )
        else:
            meta = self._metadata(run_id, start_time, seed_used, net, optimizer, setup, device, gpu_ids, restore_path)
        meta['process'] = {'pid': os.getpid(), 'host_id': _host_id()}
        meta['epochs_completed'] = len(history)
        meta['history'] = history
        write_json(meta_path, meta)
        self._live_meta = meta
        self._send_report('started')
        self._publish('resumed after ' + str(len(history)) + ' epoch(s)' if resumed_meta is not None else 'started')

        # the starting weights are validated once, before the first epoch: chance level for pre, the
        # pre-literate model for illit and lit. Kept apart from cat_scores and valloss, which continue the
        # pre-literate model's, and saved with the first checkpoint, so a run stopped before that leaves
        # nothing to resume
        initial = None
        if start_epoch == 0:
            initial = self._initial_validation(
                net, setup.validation, device, criterion, setup.cat_scores.shape[1], meta, meta_path
            )

        for epoch in range(start_epoch, setup.max_epochs):
            summary = self._train_epoch(net, setup, device, criterion, optimizer, epoch, start_time)
            history.append(summary)
            self._save_checkpoint(net, optimizer, setup, history, summary['epoch'], initial)
            initial = None
            meta['epochs_completed'] = len(history)
            meta['history'] = history
            write_json(meta_path, meta)
            self._live = None
            self._send_report('epoch_end')
            self._publish('epoch ' + str(summary['epoch']) + ' (' + self._accuracy_text(summary) + ')')

        logger.info('finished in %s', duration_text(time.time() - start_time))
        self._finish(meta, meta_path, history, setup)

    def _restore(self, restore_path: str, net: nn.Module, optimizer, setup: _Setup) -> tuple[int, list]:
        """Load a checkpoint into the run; returns the epoch to continue at (counted in this run) and the history."""
        logger.info('restoring from %s', restore_path)
        ckpt = torch.load(restore_path, map_location='cpu', weights_only=True)
        _unwrap(net).load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'cat_scores' in ckpt:
            prev = ckpt['cat_scores'].numpy()
            r, c = min(prev.shape[0], setup.cat_scores.shape[0]), min(prev.shape[1], setup.cat_scores.shape[1])
            setup.cat_scores[:r, :c] = prev[:r, :c]
        if 'trainloss' in ckpt:
            setup.trainloss = list(ckpt['trainloss'])
        if 'valloss' in ckpt:
            setup.valloss = list(ckpt['valloss'])
        if 'rng' in ckpt:
            torch.set_rng_state(ckpt['rng'].cpu().to(torch.uint8))
        self._restore_extra(ckpt)
        start_epoch = ckpt['epoch'] - setup.shift_epoch + 1
        logger.info('resuming at epoch %d of %d', setup.shift_epoch + start_epoch, setup.shift_epoch + setup.max_epochs)
        return start_epoch, list(ckpt.get('history', []))

    def _train_epoch(
        self, net: nn.Module, setup: _Setup, device: torch.device, criterion, optimizer, epoch: int, start_time: float
    ) -> dict[str, Any]:
        """One pass over the train set, then the validation of the weights after it; returns its history entry."""
        epoch_start = time.time()
        epoch_lr = optimizer.param_groups[0]['lr']
        # loss, correct predictions and count of the epoch's images and words
        train_sums = np.zeros(6)
        gc.collect()
        logger.info('epoch %d', setup.shift_epoch + epoch)
        self._before_epoch(net, epoch)
        net.train()
        batches = len(setup.training)
        for batch_i, (images, labels) in enumerate(tqdm(setup.training, desc='Training', total=batches)):
            self._report_progress('training', batch_i, batches)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            pred = net(images)[-1]
            loss = criterion(pred, labels)
            setup.trainloss.append(loss.item())
            train_sums += batch_sums(pred, labels, self.img_classes)
            if batch_i % 100 == 0:
                logger.debug('epoch %d, batch %d, training loss %.4f', setup.shift_epoch + epoch, batch_i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_sums, val_extra = self._validate_epoch(
            net, setup.validation, device, criterion, setup.cat_scores[setup.shift_epoch + epoch], setup.valloss, epoch
        )
        self._log_eta(start_time, epoch_start, setup.shift_epoch + epoch, epoch + 1, setup.max_epochs)
        return {
            'epoch': setup.shift_epoch + epoch,
            'lr': epoch_lr,
            **means('train', train_sums),
            **means('val', val_sums),
            **val_extra,
            'duration_seconds': round(time.time() - epoch_start, 1),
            'finished_at': now_iso(),
        }

    def _log_eta(self, start_time: float, epoch_start: float, epoch: int, epochs_done: int, max_epochs: int) -> None:
        """Log the time of the epoch and the estimated time left, and append them to ``training_eta_log.txt``."""
        elapsed = time.time() - start_time
        remaining = elapsed / epochs_done * (max_epochs - epochs_done)
        lines = [
            'END OF EPOCH ' + str(epoch),
            'epoch time: ' + duration_text(time.time() - epoch_start),
            'elapsed: ' + duration_text(elapsed),
            'estimated remaining: ' + duration_text(remaining),
            'estimated total: ' + duration_text(elapsed + remaining),
        ]
        logger.info('; '.join(lines))
        with open(os.path.join(self.model_dir, 'training_eta_log.txt'), 'a') as f:
            f.write('=' * 80 + '\n' + '\n'.join(lines) + '\n' + '=' * 80 + '\n\n')

    def _save_checkpoint(
        self, net: nn.Module, optimizer, setup: _Setup, history: list, epoch: int, initial: tuple | None
    ) -> None:
        """Write the epoch's checkpoint and the score arrays; ``initial`` once, with the first checkpoint."""
        ckpt = {
            'epoch': epoch,
            'state_dict': _unwrap(net).state_dict(),
            'optimizer': optimizer.state_dict(),
            'cat_scores': torch.from_numpy(setup.cat_scores),
            'trainloss': setup.trainloss,
            'valloss': setup.valloss,
            'rng': torch.get_rng_state(),
            'history': history,
            **self._checkpoint_extra(),
        }
        path = os.path.join(self.model_dir, 'checkpoints', 'epoch_%02d.pth.tar' % epoch)
        torch.save(ckpt, path + '.part')
        os.replace(path + '.part', path)
        np.save(os.path.join(self.model_dir, 'cat_scores.npy'), setup.cat_scores)
        np.save(os.path.join(self.model_dir, 'trainloss.npy'), np.array(setup.trainloss))
        np.save(os.path.join(self.model_dir, 'valloss.npy'), np.array(setup.valloss))
        if initial is not None:
            np.save(os.path.join(self.model_dir, 'initial_cat_scores.npy'), initial[0])
            np.save(os.path.join(self.model_dir, 'initial_valloss.npy'), np.array(initial[1]))
        self._save_extra()
        logger.info('saved %s', path)

    def _finish(self, meta: dict, meta_path: str, history: list, setup: _Setup) -> None:
        """Once the last epoch is done: the model file is the last checkpoint, under the model's name."""
        last_epoch = setup.shift_epoch + setup.max_epochs - 1
        last_ckpt = os.path.join(self.model_dir, 'checkpoints', 'epoch_%02d.pth.tar' % last_epoch)
        if not (history and history[-1]['epoch'] == last_epoch and os.path.isfile(last_ckpt)):
            return
        model_file = os.path.join(self.model_dir, self.model_name + '.pth.tar')
        shutil.copyfile(last_ckpt, model_file)
        meta['status'] = 'complete'
        meta['finished_at'] = now_iso()
        # summed over epochs, so time between interrupted runs isn't counted
        meta['training_seconds'] = round(sum(h['duration_seconds'] for h in history), 1)
        meta['model_file_sha256'] = sha256_file(model_file)
        write_json(meta_path, meta)
        self._send_report('complete')
        logger.info('saved %s', model_file)
        self._publish('complete (final ' + self._accuracy_text(history[-1]) + ')')

    # --- validation -----------------------------------------------------------------------------

    def _validate(
        self,
        net: nn.Module,
        loader: data.DataLoader,
        device: torch.device,
        criterion,
        row: np.ndarray,
        losses: list,
        phase: str = 'validation',
        desc: str = 'Validation',
        on_batch: Callable | None = None,
    ) -> np.ndarray:
        """One pass over the val set without gradients.

        Writes the accuracy of val batch ``k`` (= class ``k``) to ``row[k]``, appends each batch's loss to
        ``losses`` and returns the summed :func:`~corread.training.metrics.batch_sums`.
        ``on_batch(h, labels)``, if given, gets each batch's dense-layer activations and labels.
        """
        sums = np.zeros(6)
        # eval mode makes BatchNorm (CORnet-S) use its running statistics; in train mode each
        # single-class val batch would be normalised by its own statistics, removing most of the class signal
        net.eval()
        with torch.no_grad():
            for k, (images, labels) in enumerate(tqdm(loader, desc=desc, total=len(loader))):
                self._report_progress(phase, k, len(loader))
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                *_, h, pred = net(images)
                if on_batch is not None:
                    on_batch(h, labels)
                row[k] = accuracy(pred, labels)
                losses.append(criterion(pred, labels).item())
                sums += batch_sums(pred, labels, self.img_classes)
        return sums

    def _initial_validation(
        self, net, loader, device, criterion, classes: int, meta: dict, meta_path: str
    ) -> tuple[np.ndarray, list]:
        """Validate the starting weights into ``meta['initial_validation']``; returns their scores and losses."""
        start = time.time()
        initial: tuple[np.ndarray, list] = (np.zeros(classes), [])
        sums = self._validate(
            net,
            loader,
            device,
            criterion,
            initial[0],
            initial[1],
            phase='initial_validation',
            desc='Initial validation',
        )
        meta['initial_validation'] = {
            **means('val', sums),
            'duration_seconds': round(time.time() - start, 1),
            'finished_at': now_iso(),
        }
        logger.info('initial validation: %s', meta['initial_validation'])
        write_json(meta_path, meta)
        return initial

    def _validate_epoch(self, net, loader, device, criterion, row, losses, epoch: int) -> tuple[np.ndarray, dict]:
        """Validate the weights after an epoch; returns the summed batch sums and further history fields."""
        return self._validate(net, loader, device, criterion, row, losses), {}

    # --- hooks for subclasses -------------------------------------------------------------------

    def _before_epoch(self, net: nn.Module, epoch: int) -> None:
        """Called at the start of every epoch (counted from 0 in this run), before its training."""

    def _checkpoint_extra(self) -> dict[str, Any]:
        """What a checkpoint holds besides the weights, scores and history (tensors only)."""
        return {}

    def _restore_extra(self, ckpt: dict) -> None:
        """Restore what :meth:`_checkpoint_extra` saved."""

    def _save_extra(self) -> None:
        """Write the files that go to disk with every checkpoint, besides the scores."""

    # --- reports and export ---------------------------------------------------------------------

    def _payload(
        self, meta: dict, event: str, status: str, live: tuple | None = None, error: BaseException | None = None
    ) -> dict[str, Any]:
        return reporting.training_payload(meta, self._group(), event, status, live, error)

    def _send_report(self, event: str, status: str | None = None, error: BaseException | None = None) -> None:
        # never raises: an hours-long run should not stop over its status report
        if self._reporter is None or self._live_meta is None:
            return
        try:
            meta = self._live_meta
            self._reporter.send(self._payload(meta, event, status or meta['status'], self._live, error))
        except Exception as e:
            logger.warning('could not build the status report (%s)', e)

    def _report_progress(self, phase: str, batch: int, batches: int) -> None:
        self._live = (phase, batch, batches)
        if self._reporter is not None and self._reporter.due():
            self._send_report('progress')

    def report_status(self) -> None:
        """Send this model's current status to the reporting API now; raises if it doesn't get through.

        Works for any run in this model's folder: finished, interrupted, or training in another process.
        A run left ``in progress`` by a process that no longer exists on this machine is reported as
        ``stopped``.
        """
        url, token, _ = reporting.settings()
        if self._live_meta is not None:
            meta, live, status = self._live_meta, self._live, self._live_meta['status']
        else:
            meta_path = os.path.join(self.model_dir, 'metadata.json')
            if not os.path.isfile(meta_path):
                raise FileNotFoundError('No run to report: ' + meta_path + ' does not exist')
            meta = read_json(meta_path)
            live, status = None, meta.get('status')
            process = meta.get('process') or {}
            if status == 'in progress' and process.get('host_id') == _host_id() and not _still_running(process):
                status = 'stopped'
        reporting.post(url, token, self._payload(meta, 'snapshot', status, live))

    def _publish(self, message: str, status: str | None = None) -> None:
        """Commit the run's metadata (and once complete its files) to its Hub repo (``export=True``).

        A failed upload only warns, since the next one carries the whole metadata.
        """
        if self._export_to is None:
            return
        repo_id = self._export_to[0]
        try:
            publish_model(repo_id, self.model_dir, self._group(), message, status=status)
            logger.info('exported to https://huggingface.co/%s', repo_id)
        except Exception as e:
            logger.warning('could not export to %s (%s); the next export will catch up', repo_id, e)

    def _stop_export(self, status: str) -> None:
        """After the run stopped early: record the status on the Hub, or delete the repo if nothing is left."""
        if self._export_to is None:
            return
        repo_id, run_id = self._export_to
        try:
            if os.path.isfile(os.path.join(self.model_dir, 'metadata.json')) and not self._nothing_to_resume():
                self._publish(status, status=status)
            else:
                discard_model_export(repo_id, run_id)
        except Exception as e:
            logger.warning('could not update %s on the Hub (%s)', repo_id, e)

    @staticmethod
    def _accuracy_text(scores: dict) -> str:
        text = 'val acc images %.1f%%' % scores['val_accuracy_images']
        if scores.get('val_accuracy_words') is not None:
            text += ', words %.1f%%' % scores['val_accuracy_words']
        return text
