import json
import os

import numpy as np
import pytest
import torch

from corread import FamilyTrainingInstance, TrainingInstance
from corread.training import FAMILY_MODES

from .conftest import TINY


def load(folder, name):
    return torch.load(os.path.join(folder, name + '.pth.tar'), map_location='cpu', weights_only=True)


def test_family_trains_every_mode(in_trained):
    family = FamilyTrainingInstance(model_choice='z', wordset='words', **TINY)
    assert list(family.instances) == list(FAMILY_MODES)
    for mode, instance in family.instances.items():
        assert instance.is_complete(), mode
        meta = json.load(open(os.path.join(instance.model_dir, 'metadata.json')))
        assert meta['status'] == 'complete' and meta['mode'] == mode
        assert len(meta['history']) == 2 and meta['initial_validation'] is not None
        assert meta['environment']['code_sha256']
        assert 'host' not in meta['process'] and meta['process']['host_id']
        assert os.path.isfile(os.path.join(instance.model_dir, instance.model_name + '.pth.tar'))
    # a family run again finds every model trained and trains nothing
    family.train()


def test_folders_and_epoch_numbering(in_trained):
    models = os.path.join(in_trained, 'models')
    assert sorted(os.listdir(models)) == ['illiterate', 'pre', 'words']
    literate = json.load(open(os.path.join(models, 'words', 'lit_bias_z_3', 'metadata.json')))
    assert literate['epochs'] == {**literate['epochs'], 'first': 2, 'last': 3, 'count': 2}
    assert [h['epoch'] for h in literate['history']] == [2, 3]
    assert literate['initialised_from']['name'] == 'pre_z_1'
    assert literate['data']['wordset']['images_digest']
    assert literate['model']['class'] == 'CORnetZBiasedWords' and literate['model']['outputs'] == 2000
    # the literate run continues the pre-literate model's scores and losses
    pre = os.path.join(models, 'pre', 'pre_z_1')
    lit = os.path.join(models, 'words', 'lit_bias_z_3')
    cat_pre, cat_lit = np.load(os.path.join(pre, 'cat_scores.npy')), np.load(os.path.join(lit, 'cat_scores.npy'))
    assert cat_lit.shape == (4, 2000)
    assert np.array_equal(cat_lit[:2, :1000], cat_pre)
    trainloss_pre = np.load(os.path.join(pre, 'trainloss.npy'))
    assert np.array_equal(np.load(os.path.join(lit, 'trainloss.npy'))[: len(trainloss_pre)], trainloss_pre)


def test_initial_validation_of_a_literate_run_is_the_pre_literate_model(in_trained):
    models = os.path.join(in_trained, 'models')
    pre = json.load(open(os.path.join(models, 'pre', 'pre_z_1', 'metadata.json')))
    illit = json.load(open(os.path.join(models, 'illiterate', 'illit_z_3', 'metadata.json')))
    assert illit['initial_validation']['val_accuracy_images'] == pytest.approx(
        pre['history'][-1]['val_accuracy_images']
    )


def test_resumed_run_equals_an_uninterrupted_one(in_trained, monkeypatch):
    kwargs = dict(TINY, mode='lit_no_bias', wordset='words', model_choice='z')
    TrainingInstance(name='straight', **kwargs).train()

    original = TrainingInstance._before_epoch

    def interrupt(self, net, epoch):
        if epoch == 1:
            raise KeyboardInterrupt
        original(self, net, epoch)

    monkeypatch.setattr(TrainingInstance, '_before_epoch', interrupt)
    run = TrainingInstance(name='resumed', **kwargs)
    with pytest.raises(KeyboardInterrupt):
        run.train()
    assert json.load(open(os.path.join(run.model_dir, 'metadata.json')))['status'] == 'interrupted'
    with pytest.raises(FileExistsError, match='resume=True'):
        TrainingInstance(name='resumed', **kwargs).train()
    monkeypatch.setattr(TrainingInstance, '_before_epoch', original)
    TrainingInstance(name='resumed', resume=True, **kwargs).train()

    straight = load(os.path.join(in_trained, 'models', 'straight', 'lit_no_bias_z_3'), 'lit_no_bias_z_3')
    resumed = load(run.model_dir, 'lit_no_bias_z_3')
    for key, value in straight['state_dict'].items():
        assert torch.equal(value, resumed['state_dict'][key]), key
    assert straight['trainloss'] == resumed['trainloss']


def test_resume_refuses_changed_arguments(in_trained, monkeypatch):
    kwargs = dict(TINY, mode='illit', name='changing', model_choice='z')
    original = TrainingInstance._before_epoch

    def interrupt(self, net, epoch):
        if epoch == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(TrainingInstance, '_before_epoch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        TrainingInstance(**kwargs).train()
    monkeypatch.setattr(TrainingInstance, '_before_epoch', original)
    with pytest.raises(ValueError, match=r'lr was 0\.01, now 0\.1'):
        TrainingInstance(resume=True, **dict(kwargs, lr=0.1)).train()


def test_interrupt_before_the_first_epoch_leaves_nothing(in_trained, monkeypatch):
    def interrupt(self, net, epoch):
        raise KeyboardInterrupt

    monkeypatch.setattr(TrainingInstance, '_before_epoch', interrupt)
    run = TrainingInstance(mode='illit', name='nothing', model_choice='z', **TINY)
    with pytest.raises(KeyboardInterrupt):
        run.train()
    assert not os.path.exists(run.model_dir)


def test_arguments_are_checked_before_training(in_trained):
    with pytest.raises(ValueError, match='mode'):
        TrainingInstance(mode='literate')
    with pytest.raises(ValueError, match='model_choice'):
        TrainingInstance(model_choice='x')
    with pytest.raises(ValueError, match='word set'):
        TrainingInstance(mode='lit_bias', **TINY)
    with pytest.raises(ValueError, match='either wordset or wrd_path'):
        TrainingInstance(mode='lit_bias', wordset='words', wrd_path='x', **TINY)
    with pytest.raises(ValueError, match='plain folder name'):
        TrainingInstance(name='../escape')
    with pytest.raises(TypeError, match='no mode'):
        FamilyTrainingInstance(mode='pre')
    with pytest.raises(ValueError, match='different arguments'):
        TrainingInstance(mode='pre', **dict(TINY, lr=0.5)).is_complete()


def test_checkpoints_load_without_pickled_code(in_trained):
    folder = os.path.join(in_trained, 'models', 'pre', 'pre_z_1')
    checkpoint = torch.load(os.path.join(folder, 'checkpoints', 'epoch_01.pth.tar'), weights_only=True)
    assert set(checkpoint) >= {'epoch', 'state_dict', 'optimizer', 'cat_scores', 'trainloss', 'valloss', 'rng'}
