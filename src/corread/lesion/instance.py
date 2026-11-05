# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Lesion runs: dense-layer units of a trained literate CORnet-Z are lesioned, then the model recovers."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from .. import reporting
from .._io import check_name, model_parts, now_iso, read_json, write_json
from .._provenance import sha256_file
from ..hub._client import read_sidecar
from ..models import DENSE_UNITS, build_network
from ..paths import get_path
from ..training import descriptions
from ..training.data import image_dataset, word_dataset
from ..training.instance import LITERATE_MODES, TrainingInstance, _Setup, _unwrap
from ..training.metrics import means
from .units import UnitSums, _whole, check_units, lesion, selective, selectivity, word_channel

logger = logging.getLogger(__name__)

STAGES = ('intact', 'lesioned')
"""The rows of the unit tracking before the recovery epochs."""
TRACKED = ('mean_words', 'mean_images', 'sd_images', 'selectivity', 'mean_val_images')
"""What the unit tracking records of every unit at each row."""


def _read_source(models_dir: str, model: str) -> tuple[str, str, str, dict, str]:
    """``(group, name, folder, metadata, model file)`` of a finished literate CORnet-Z model."""
    group, name = model_parts(model)
    folder = os.path.join(models_dir, group, name)
    meta_path = os.path.join(folder, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError('Not a model folder (no metadata.json): ' + folder)
    meta = read_json(meta_path)
    model = group + '/' + name
    if meta.get('status') != 'complete':
        raise ValueError(model + ' has not finished training (status: ' + str(meta.get('status')) + ')')
    if meta.get('mode') not in LITERATE_MODES:
        raise ValueError(
            'Lesions need a literate model ('
            + ' or '.join(LITERATE_MODES)
            + '); '
            + model
            + ' is '
            + repr(meta.get('mode'))
        )
    choice = (meta.get('arguments') or {}).get('model_choice')
    if choice != 'z':
        raise NotImplementedError(
            'Lesions are implemented for CORnet-Z only; ' + model + ' is CORnet-' + str(choice).upper()
        )
    model_file = os.path.join(folder, name + '.pth.tar')
    if not os.path.isfile(model_file):
        raise FileNotFoundError('Model file not found: ' + model_file)
    return group, name, folder, meta, model_file


def _source_net(meta: dict, state_dict: dict) -> nn.Module:
    """The network of a literate model, from its metadata and weights."""
    args = meta['arguments']
    net = build_network('z', meta['mode'], args['img_classes'], args['wrd_classes'])
    net.load_state_dict(state_dict)
    return net


class LesionInstance(TrainingInstance):
    """Lesion dense-layer units of a finished literate CORnet-Z model, then let the model recover.

    A lesion re-initialises a unit's incoming weights (its IT conv filter and bias) once, as PyTorch
    initialises an untrained layer; afterwards they train like every other weight. The model then keeps
    training the way its literate training did (same data, transforms, SGD settings and validation)
    for ``max_epochs`` epochs.

    Before the first epoch the run validates and measures the intact model, lesions it, then validates
    and measures the lesioned one; every epoch then trains, validates and measures. To *measure* is to
    record each unit's mean activation to all val words and val images, its mean and SD over the
    selectivity sample, and from these its word selectivity (see :meth:`word_selectivity`), into
    ``unit_tracking.npz``.

    The run goes into ``<models>/<group>/<name>/``, next to the lesioned model (by default
    ``lesion_<mode>_z_<units>u_<last epoch>``), and its epochs continue that model's numbering.

    Parameters
    ----------
    model : str
        The model to lesion, as ``'<group>/<name>'`` in the models folder, e.g.
        ``'english/lit_bias_z_79'``.
    units : list of int
        Dense-layer units to lesion (0..511), e.g. ``word_selectivity(model)['selective_units']``.
        ``[]`` trains the same way without a lesion, as a control.
    name : str, optional
        Folder name of the run.
    reset_outgoing : bool
        Also re-initialise the units' weights in the readout.
    max_epochs : int
        Recovery epochs.
    silent_epochs : int
        Silence the units (multiply their output by 0, so they neither drive the readout nor learn) for
        the first ``silent_epochs`` recovery epochs, and re-initialise them when they are released;
        ``max_epochs`` makes the lesion permanent.
    threshold, selectivity_images, selectivity_seed
        Word selectivity, as for :meth:`word_selectivity`.
    batch_size, lr, momentum, weight_decay : optional
        By default the lesioned model's.

    The remaining parameters are as for :class:`~corread.training.TrainingInstance`. Hub export is
    not available for lesion runs.
    """

    def __init__(
        self,
        model: str,
        units: list[int],
        name: str | None = None,
        *,
        reset_outgoing: bool = False,
        max_epochs: int = 10,
        silent_epochs: int = 0,
        threshold: float = 3,
        selectivity_images: int = 5000,
        selectivity_seed: int = 0,
        seed: int | None = None,
        resume: bool = False,
        report: bool = False,
        models_dir: str | None = None,
        img_path: str | None = None,
        wrd_path: str | None = None,
        num_workers: int = 1,
        gpus: int | list[int] | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        momentum: float | None = None,
        weight_decay: float | None = None,
    ):
        if name is not None:
            check_name(name)
        if not isinstance(reset_outgoing, bool):
            raise ValueError('reset_outgoing must be True or False, got: ' + repr(reset_outgoing))
        if not _whole(max_epochs, 1):
            raise ValueError('max_epochs must be a whole number of at least 1, got: ' + repr(max_epochs))
        if not _whole(silent_epochs, 0) or silent_epochs > max_epochs:
            raise ValueError(
                'silent_epochs must be a whole number from 0 to max_epochs ('
                + str(max_epochs)
                + '), got: '
                + repr(silent_epochs)
            )
        if not _whole(selectivity_images, 2):
            raise ValueError(
                'selectivity_images must be a whole number of at least 2 (for an SD), got: ' + repr(selectivity_images)
            )
        if not _whole(selectivity_seed, 0):
            raise ValueError('selectivity_seed must be a whole number of at least 0, got: ' + repr(selectivity_seed))
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.number)):
            raise ValueError('threshold must be a number, got: ' + repr(threshold))
        if batch_size is not None and not _whole(batch_size, 1):
            raise ValueError('batch_size must be a whole number of at least 1, got: ' + repr(batch_size))

        models_dir = models_dir if models_dir is not None else get_path('models')
        self.source_group, self.source_name, self.source_dir, self.source_meta, self.source_file = _read_source(
            models_dir, model
        )
        self.model = self.source_group + '/' + self.source_name
        self.source_last_epoch = self.source_meta['epochs']['last']
        source_args = self.source_meta['arguments']
        self.units = check_units(units)
        self.reset_outgoing = reset_outgoing
        self.max_epochs = int(max_epochs)
        self.silent_epochs = int(silent_epochs)
        self.threshold = float(threshold)
        self.selectivity_images = int(selectivity_images)
        self.selectivity_seed = int(selectivity_seed)

        self.mode = 'lesion'
        self.model_choice = 'z'
        self.name = name
        self.pre_name = None
        self.restore_path = None
        # the word set the model was trained on, by its name unless a path is given
        self.wordset = source_args.get('wordset') if wrd_path is None else None
        if wrd_path is not None:
            self.wrd_path = wrd_path
        elif self.wordset:
            self.wrd_path = os.path.join(get_path('wordsets'), self.wordset)
        else:
            self.wrd_path = source_args.get('wrd_path')
        self.batch_size = int(batch_size) if batch_size is not None else source_args['batch_size']
        self.lr = lr if lr is not None else source_args['lr']
        self.momentum = momentum if momentum is not None else source_args['momentum']
        self.weight_decay = weight_decay if weight_decay is not None else source_args['weight_decay']
        # these must match the lesioned network and its val set
        self.num_val_items = source_args['num_val_items']
        self.img_classes = source_args['img_classes']
        self.wrd_classes = source_args['wrd_classes']
        self._init_run(
            resume=resume,
            export=False,
            report=report,
            seed=seed,
            models_dir=models_dir,
            img_path=img_path,
            num_workers=num_workers,
            gpus=gpus,
        )
        # while training: the selectivity sample's loader, the unit tracking (TRACKED arrays: one row per
        # STAGES entry, then per epoch), the intact validation's scores until they are saved, and the
        # hook that silences the lesioned units while they are silent
        self._sample_gen: data.DataLoader | None = None
        self._tracking: dict[str, np.ndarray] | None = None
        self._intact: tuple[np.ndarray, list] | None = None
        self._silencer: torch.utils.hooks.RemovableHandle | None = None

    @staticmethod
    def word_selectivity(
        model: str,
        threshold: float = 3,
        selectivity_images: int = 5000,
        selectivity_seed: int = 0,
        gpus: int | list[int] | None = None,
        num_workers: int = 1,
        batch_size: int = 100,
        models_dir: str | None = None,
        img_path: str | None = None,
        wrd_path: str | None = None,
    ) -> dict[str, Any]:
        """How word-selective each dense-layer unit of a literate model is.

        For unit ``u``: ``selectivity[u]`` = (mean activation to the words − mean activation to the images)
        / SD of its activations to the images, where the words are all val word images of the model's word
        set and the images the *selectivity sample* (``selectivity_images`` ImageNet train images drawn with
        ``selectivity_seed``), both through the fixed validation view. A unit is word-selective when
        ``selectivity[u] >= threshold``.

        Returns
        -------
        dict
            ``mean_words``, ``mean_images``, ``sd_images``, ``selectivity``, ``word_selective``: arrays over
            the 512 units; ``selective_units``: the word-selective units, most selective first; ``ranking``:
            all units, most selective first; ``word_channel``: the units the biased model's word outputs
            read, else ``None``; and ``model``, ``threshold``, ``num_words``, ``num_images``,
            ``selectivity_seed``.
        """
        probe = LesionInstance(
            model,
            units=[],
            threshold=threshold,
            selectivity_images=selectivity_images,
            selectivity_seed=selectivity_seed,
            gpus=gpus,
            num_workers=num_workers,
            batch_size=batch_size,
            models_dir=models_dir,
            img_path=img_path,
            wrd_path=wrd_path,
        )
        return probe._model_selectivity()

    def _model_selectivity(self) -> dict[str, Any]:
        device = self._device(self._gpu_ids())
        state_dict = torch.load(self.source_file, map_location='cpu', weights_only=True)['state_dict']
        net = _source_net(self.source_meta, state_dict)
        net.to(device)
        words = word_dataset(self.wrd_path, 'val', label_offset=self.img_classes)
        word_gen = self._loader(words, self.batch_size, False, device)
        sample_gen, _ = self._sample_loader(device)
        responses = UnitSums(self.img_classes)
        self._pass(net, word_gen, device, responses, 'words', 'Val words')
        self._pass(net, sample_gen, device, responses, 'images', 'Selectivity sample')

        mean_words, mean_images, sd_images = responses.mean('words'), responses.mean('images'), responses.sd('images')
        values = selectivity(mean_words, mean_images, sd_images)
        word_selective = selective(values, self.threshold)
        ranking = [int(u) for u in np.argsort(-values, kind='stable')]  # nan sorts last
        return {
            'model': self.model,
            'mean_words': mean_words,
            'mean_images': mean_images,
            'sd_images': sd_images,
            'selectivity': values,
            'word_selective': word_selective,
            'selective_units': [u for u in ranking if word_selective[u]],
            'ranking': ranking,
            'word_channel': word_channel(net),
            'threshold': self.threshold,
            'num_words': responses.count('words'),
            'num_images': responses.count('images'),
            'selectivity_seed': self.selectivity_seed,
        }

    # --- where the run goes, and what it was run with -------------------------------------------

    def _output_location(self) -> tuple[str, str]:
        last_epoch = self.source_last_epoch + self.max_epochs
        model_name = self.name or (
            'lesion_'
            + self.source_meta['mode']
            + '_z_'
            + str(len(self.units))
            + 'u_'
            + ('silent' + str(self.silent_epochs) + '_' if self.silent_epochs else '')
            + str(last_epoch)
        )
        if model_name == self.source_name:
            raise ValueError('name ' + repr(model_name) + " is the lesioned model's own; choose another name")
        return os.path.join(self.models_dir, self.source_group, model_name), model_name

    def _arguments(self) -> dict[str, Any]:
        return {
            'model': self.model,
            'units': self.units,
            'name': self.name,
            'mode': self.mode,
            'model_choice': self.model_choice,
            'reset_outgoing': self.reset_outgoing,
            'max_epochs': self.max_epochs,
            'silent_epochs': self.silent_epochs,
            'threshold': self.threshold,
            'selectivity_images': self.selectivity_images,
            'selectivity_seed': self.selectivity_seed,
            'wordset': self.wordset,
            'wrd_path': os.path.abspath(self.wrd_path) if self.wrd_path else None,
            'img_path': os.path.abspath(self.img_path),
            'models_dir': os.path.abspath(self.models_dir),
            'resume': self.resume,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'num_val_items': self.num_val_items,
            'img_classes': self.img_classes,
            'wrd_classes': self.wrd_classes,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
        }

    # --- the network and data of the run --------------------------------------------------------

    def _build(self, device: torch.device) -> _Setup:
        if self.wrd_path is None:
            raise ValueError('The word set of ' + self.model + ' is not known; pass wrd_path.')
        logger.info('building the lesion run of %s', self.model)
        training, validation, data_info = self._literate_loaders(device)
        self._sample_gen, data_info['selectivity_images'] = self._sample_loader(device)

        # scores and losses continue the lesioned model's
        shift_epoch = self.source_last_epoch + 1
        cat_scores = np.zeros((shift_epoch + self.max_epochs, self.img_classes + self.wrd_classes))
        cat_scores[:shift_epoch] = np.load(os.path.join(self.source_dir, 'cat_scores.npy'))[:shift_epoch]
        trainloss = np.load(os.path.join(self.source_dir, 'trainloss.npy')).tolist()
        valloss = np.load(os.path.join(self.source_dir, 'valloss.npy')).tolist()

        ckpt = torch.load(self.source_file, map_location='cpu', weights_only=True)
        if ckpt['epoch'] != self.source_last_epoch:
            raise ValueError(
                self.model
                + "'s model file is from epoch "
                + str(ckpt['epoch'])
                + ', but its metadata.json ends at epoch '
                + str(self.source_last_epoch)
            )
        net = _source_net(self.source_meta, ckpt['state_dict'])
        self._tracking = {k: np.full((len(STAGES) + self.max_epochs, DENSE_UNITS), np.nan) for k in TRACKED}
        return _Setup(
            net,
            training,
            validation,
            data_info,
            self._source_info(ckpt),
            cat_scores,
            trainloss,
            valloss,
            self.max_epochs,
            shift_epoch,
        )

    def _sample_loader(self, device: torch.device) -> tuple[data.DataLoader, dict[str, Any]]:
        """The selectivity sample, through the fixed validation view; its loader and its description."""
        images = image_dataset(self.img_path, 'train', augment=False)
        if self.selectivity_images > len(images):
            raise ValueError(
                'selectivity_images='
                + str(self.selectivity_images)
                + ', but '
                + os.path.join(self.img_path, 'train')
                + ' has only '
                + str(len(images))
                + ' images'
            )
        generator = torch.Generator().manual_seed(self.selectivity_seed)
        indices = torch.randperm(len(images), generator=generator)[: self.selectivity_images].sort().values.tolist()
        loader = self._loader(data.Subset(images, indices), self.batch_size, False, device)
        return loader, {
            'path': os.path.abspath(os.path.join(self.img_path, 'train')),
            'images': len(indices),
            'drawn_from': len(images),
            'seed': self.selectivity_seed,
            'drawing': 'torch.randperm(drawn_from, generator=torch.Generator().manual_seed(seed))[:images], '
            'indices into the train images in ImageFolder order (sorted folders)',
            'transform': str(images.transform),
        }

    def _source_info(self, ckpt: dict) -> dict[str, Any]:
        """Which model the run lesions, for ``metadata.json`` (``initialised_from``)."""
        info = {
            'model': self.model,
            'group': self.source_group,
            'name': self.source_name,
            'mode': self.source_meta.get('mode'),
            'path': os.path.abspath(self.source_dir),
            'file_sha256': sha256_file(self.source_file),
            'epoch': ckpt['epoch'],
            'metadata_sha256': sha256_file(os.path.join(self.source_dir, 'metadata.json')),
            'run_id': self.source_meta.get('run_id'),
            'seed_used': self.source_meta.get('seed_used'),
            'arguments': self.source_meta.get('arguments'),
        }
        hub = read_sidecar(self.source_dir)
        if hub is not None:
            info['hub'] = hub
        return info

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
        meta = super()._metadata(run_id, start_time, seed_used, net, optimizer, setup, device, gpu_ids, restore_path)
        channel = word_channel(_unwrap(net))
        meta['epochs']['numbering'] = descriptions.LESION_EPOCH_NUMBERING
        meta['reproduce'] = descriptions.LESION_REPRODUCE
        meta['model']['outputs'] = self.img_classes + self.wrd_classes
        meta['model']['output_layout'] = 'image classes 0..img_classes-1, then word classes'
        meta['lesion'] = {
            'model': self.model,
            'layer': descriptions.LESION_LAYER.format(units=DENSE_UNITS, last=DENSE_UNITS - 1),
            'units': self.units,
            'count': len(self.units),
            'units_total': DENSE_UNITS,
            'word_channel': list(channel) if channel else None,
            'word_channel_meaning': descriptions.WORD_CHANNEL_MEANING,
            'lesioned_in_word_channel': sum(channel[0] <= u <= channel[1] for u in self.units) if channel else None,
            'method': descriptions.lesion_method(
                descriptions.silence(self.silent_epochs, self.max_epochs), self.silent_epochs
            ),
            'silent_epochs': self.silent_epochs,
            'released_at': None,
            'reset_outgoing': self.reset_outgoing,
            'outgoing': (
                descriptions.outgoing_reset(channel[0] if channel else None)
                if self.reset_outgoing
                else descriptions.OUTGOING_KEPT
            ),
            'random_numbers': descriptions.LESION_RANDOM_NUMBERS,
            'applied_at': None,
            'filter_norm_before': None,
            'filter_norm_after': None,
            'filter_norm': descriptions.FILTER_NORM,
            'selectivity': {
                'definition': descriptions.SELECTIVITY_DEFINITION,
                'threshold': self.threshold,
                'words': descriptions.SELECTIVITY_WORDS,
                'images': descriptions.SELECTIVITY_IMAGES,
            },
        }
        meta['intact_validation'] = None
        meta['epoch_procedure'] = descriptions.LESION_EPOCH_PROCEDURE
        meta['metrics'] += descriptions.LESION_METRICS
        meta['cat_scores'] = descriptions.LESION_CAT_SCORES
        meta['files']['checkpoints'] += ', unit_tracking (the arrays of unit_tracking.npz)'
        meta['files'].update(descriptions.LESION_FILES)
        return meta

    # --- the lesion -----------------------------------------------------------------------------

    def _initial_validation(
        self, net, loader, device, criterion, classes: int, meta: dict, meta_path: str
    ) -> tuple[np.ndarray, list]:
        """Measure the intact model, lesion it, and measure the lesioned one, the start of the recovery."""
        start = time.time()
        intact: tuple[np.ndarray, list] = (np.zeros(classes), [])
        sums, fields = self._measure(net, loader, device, criterion, intact[0], intact[1], 0, 'intact_', ' (intact)')
        meta['intact_validation'] = {
            **means('val', sums),
            **fields,
            'duration_seconds': round(time.time() - start, 1),
            'finished_at': now_iso(),
        }
        logger.info('intact validation: %s', meta['intact_validation'])
        self._intact = intact

        model = _unwrap(net)
        if self.silent_epochs:
            # re-initialised when released, see _before_epoch
            self._silence(model)
            meta['lesion'].update(applied_at=now_iso())
            logger.info('silenced %d units for %d epochs: %s', len(self.units), self.silent_epochs, self.units)
        else:
            before, after = lesion(
                model, self.units, self.reset_outgoing, torch.Generator().manual_seed(meta['seed_used'])
            )
            meta['lesion'].update(applied_at=now_iso(), filter_norm_before=before, filter_norm_after=after)
            logger.info('lesioned %d units: %s', len(self.units), self.units)
        write_json(meta_path, meta)

        start = time.time()
        initial: tuple[np.ndarray, list] = (np.zeros(classes), [])
        sums, fields = self._measure(
            net, loader, device, criterion, initial[0], initial[1], 1, 'initial_', ' (lesioned)'
        )
        meta['initial_validation'] = {
            **means('val', sums),
            **fields,
            'duration_seconds': round(time.time() - start, 1),
            'finished_at': now_iso(),
        }
        logger.info('validation after the lesion: %s', meta['initial_validation'])
        write_json(meta_path, meta)
        return initial

    def _silence(self, model: nn.Module) -> None:
        """Silence the lesioned units of ``model`` (not a DataParallel) until they are released."""
        if self._silencer is not None:
            return
        mask = torch.ones(DENSE_UNITS)
        mask[self.units] = 0
        self._silencer = model.IT.register_forward_hook(
            lambda module, inp, out: out * mask.to(out.device, out.dtype).view(1, -1, 1, 1)
        )

    def _before_epoch(self, net: nn.Module, epoch: int) -> None:
        """With ``silent_epochs``: keep the units silent, and release and re-initialise them at epoch ``silent_epochs``.

        The weights there are always those of the end of the silence, also when the run resumed from a
        checkpoint, so they are released whenever that epoch starts.
        """
        if not self.silent_epochs or not self.units:
            return
        model = _unwrap(net)
        if epoch < self.silent_epochs:
            self._silence(model)
        elif epoch == self.silent_epochs:
            if self._silencer is not None:
                self._silencer.remove()
                self._silencer = None
            meta = self._live_meta
            before, after = lesion(
                model, self.units, self.reset_outgoing, torch.Generator().manual_seed(meta['seed_used'])
            )
            meta['lesion'].update(released_at=now_iso(), filter_norm_before=before, filter_norm_after=after)
            write_json(os.path.join(self.model_dir, 'metadata.json'), meta)
            logger.info('released %d units, re-initialised: %s', len(self.units), self.units)

    # --- measuring the units --------------------------------------------------------------------

    def _validate_epoch(self, net, loader, device, criterion, row, losses, epoch: int) -> tuple[np.ndarray, dict]:
        return self._measure(net, loader, device, criterion, row, losses, len(STAGES) + epoch)

    def _measure(
        self, net, loader, device, criterion, row, losses, r: int, prefix: str = '', label: str = ''
    ) -> tuple[np.ndarray, dict]:
        """Validate while recording each unit's mean activation, then pass the selectivity sample.

        Fills row ``r`` of the unit tracking and returns the validation's summed batch sums and the unit
        summary for its entry. ``prefix`` names the phases in the reports, ``label`` the progress bars.
        """
        val = UnitSums(self.img_classes)
        sums = self._validate(
            net,
            loader,
            device,
            criterion,
            row,
            losses,
            phase=prefix + 'validation',
            desc='Validation' + label,
            on_batch=val,
        )
        sample = UnitSums(self.img_classes)
        self._pass(net, self._sample_gen, device, sample, prefix + 'selectivity', 'Selectivity sample' + label)
        t = self._tracking
        t['mean_words'][r] = val.mean('words')
        t['mean_val_images'][r] = val.mean('images')
        t['mean_images'][r] = sample.mean('images')
        t['sd_images'][r] = sample.sd('images')
        t['selectivity'][r] = selectivity(t['mean_words'][r], t['mean_images'][r], t['sd_images'][r])
        return sums, self._unit_summary(r)

    def _pass(
        self, net: nn.Module, loader: data.DataLoader, device: torch.device, units: UnitSums, phase: str, desc: str
    ) -> None:
        """The dense layer's responses to the images of ``loader``, in eval mode without gradients."""
        net.eval()
        with torch.no_grad():
            for batch_i, (images, labels) in enumerate(tqdm(loader, desc=desc, total=len(loader))):
                self._report_progress(phase, batch_i, len(loader))
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                units(net(images)[-2], labels)

    def _unit_summary(self, r: int) -> dict[str, Any]:
        """The lesioned and the other units' mean word activation, and how many of each are word-selective."""
        t = self._tracking
        lesioned = np.zeros(DENSE_UNITS, dtype=bool)
        lesioned[self.units] = True
        words = t['mean_words'][r]
        is_selective = selective(t['selectivity'][r], self.threshold)
        return {
            'mean_word_activation_lesioned': float(words[lesioned].mean()) if lesioned.any() else None,
            'mean_word_activation_other': float(words[~lesioned].mean()) if (~lesioned).any() else None,
            'word_selective_lesioned': int(is_selective[lesioned].sum()),
            'word_selective_other': int(is_selective[~lesioned].sum()),
        }

    # --- saved with every checkpoint ------------------------------------------------------------

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {'unit_tracking': {k: torch.from_numpy(v) for k, v in self._tracking.items()}}

    def _restore_extra(self, ckpt: dict) -> None:
        if 'unit_tracking' in ckpt:
            self._tracking = {k: v.numpy().copy() for k, v in ckpt['unit_tracking'].items()}

    def _save_extra(self) -> None:
        t = self._tracking
        first = self.source_last_epoch + 1
        np.savez(
            os.path.join(self.model_dir, 'unit_tracking.npz'),
            stage=np.array(list(STAGES) + ['epoch ' + str(first + k) for k in range(self.max_epochs)]),
            epoch=np.array([self.source_last_epoch] * len(STAGES) + list(range(first, first + self.max_epochs))),
            units=np.array(self.units, dtype=np.int64),
            threshold=np.array(self.threshold),
            word_selective=selective(t['selectivity'], self.threshold),
            **t,
        )
        if self._intact is not None:
            np.save(os.path.join(self.model_dir, 'intact_cat_scores.npy'), self._intact[0])
            np.save(os.path.join(self.model_dir, 'intact_valloss.npy'), np.array(self._intact[1]))
            self._intact = None

    def _payload(self, meta, event, status, live=None, error=None) -> dict[str, Any]:
        return reporting.lesion_payload(meta, self._group(), event, status, live, error)
