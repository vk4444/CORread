import json
import os

import numpy as np
import pytest
import torch

from corread import LesionInstance
from corread.lesion import check_units, lesion, selective, selectivity, word_channel
from corread.models import CORnetZBiasedWords, CORnetZNonbiasedWords

from .conftest import TINY

RUN = dict(selectivity_images=4, batch_size=4, gpus=0, num_workers=0)


def test_lesion_reinitialises_exactly_the_chosen_units():
    torch.manual_seed(0)
    net = CORnetZBiasedWords()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    units = [0, 470, 500]
    lesion(net, units, reset_outgoing=True, generator=torch.Generator().manual_seed(1))
    after = net.state_dict()
    others = [u for u in range(512) if u not in units]
    assert torch.equal(after['IT.conv.weight'][others], before['IT.conv.weight'][others])
    assert not torch.equal(after['IT.conv.weight'][units], before['IT.conv.weight'][units])
    # outgoing: every unit feeds the image readout; 470 and 500 also the word readout (columns 7 and 37)
    assert not torch.equal(after['bilinear.lin_img.weight'][:, units], before['bilinear.lin_img.weight'][:, units])
    assert torch.equal(after['bilinear.lin_img.weight'][:, others], before['bilinear.lin_img.weight'][:, others])
    changed = [
        c
        for c in range(49)
        if not torch.equal(after['bilinear.lin_wrd.weight'][:, c], before['bilinear.lin_wrd.weight'][:, c])
    ]
    assert changed == [7, 37]
    assert torch.equal(after['bilinear.lin_img.bias'], before['bilinear.lin_img.bias'])


def test_word_channel():
    assert word_channel(CORnetZBiasedWords()) == (463, 511)
    assert word_channel(CORnetZNonbiasedWords()) is None


def test_selectivity_definition():
    values = selectivity(np.array([3.0, 1.0, 2.0, 1.0]), np.array([1.0, 1.0, 2.0, 2.0]), np.array([0.5, 1.0, 0.0, 0.0]))
    assert values[0] == 4 and values[1] == 0 and np.isnan(values[2]) and values[3] == -np.inf
    assert selective(values, 3).tolist() == [True, False, False, False]


@pytest.mark.parametrize('units', [[512], [-1], [True], ['3'], '12', [3, 3]])
def test_units_are_checked(units):
    with pytest.raises(ValueError):
        check_units(units)


def test_word_selectivity(in_trained):
    s = LesionInstance.word_selectivity('words/lit_bias_z_3', **RUN)
    assert s['selectivity'].shape == (512,)
    assert sorted(s['ranking']) == list(range(512))
    assert s['word_channel'] == (463, 511)
    assert s['num_words'] == 4 * 2 and s['num_images'] == 4
    assert all(s['word_selective'][u] for u in s['selective_units'])


def test_lesion_run(in_trained):
    run = LesionInstance(
        'words/lit_bias_z_3', units=[0, 470, 500], reset_outgoing=True, silent_epochs=1, max_epochs=2, seed=9, **RUN
    )
    assert run.model_name == 'lesion_lit_bias_z_3u_silent1_5'
    run.train()
    meta = json.load(open(os.path.join(run.model_dir, 'metadata.json')))
    assert meta['status'] == 'complete'
    assert [h['epoch'] for h in meta['history']] == [4, 5]
    assert meta['lesion']['units'] == [0, 470, 500] and meta['lesion']['lesioned_in_word_channel'] == 2
    assert meta['lesion']['released_at'] is not None
    for entry in (meta['intact_validation'], meta['initial_validation'], *meta['history']):
        assert {'mean_word_activation_lesioned', 'word_selective_other'} <= set(entry)
    # silenced for the first epoch: the lesioned units are silent in the lesioned validation
    assert meta['initial_validation']['mean_word_activation_lesioned'] == 0
    tracking = np.load(os.path.join(run.model_dir, 'unit_tracking.npz'))
    assert tracking['stage'].tolist() == ['intact', 'lesioned', 'epoch 4', 'epoch 5']
    assert tracking['selectivity'].shape == (4, 512) and not np.isnan(tracking['mean_words']).any()


def test_control_run_without_lesion(in_trained):
    run = LesionInstance('words/lit_no_bias_z_3', units=[], max_epochs=1, seed=3, **RUN)
    run.train()
    meta = json.load(open(os.path.join(run.model_dir, 'metadata.json')))
    assert meta['intact_validation']['val_loss'] == pytest.approx(meta['initial_validation']['val_loss'])
    assert meta['lesion']['word_channel'] is None


def test_only_literate_models_can_be_lesioned(in_trained):
    with pytest.raises(ValueError, match='literate model'):
        LesionInstance('pre/pre_z_1', units=[1], **RUN)
    with pytest.raises(ValueError, match='silent_epochs'):
        LesionInstance('words/lit_bias_z_3', units=[1], max_epochs=1, silent_epochs=2, **RUN)


def test_training_settings_default_to_the_lesioned_model(in_trained):
    run = LesionInstance('words/lit_bias_z_3', units=[1], **RUN)
    assert run.lr == 0.01 and run.num_val_items == TINY['num_val_items']
