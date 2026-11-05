"""The Hub code without a network: settings, recipe validation, and the generated cards."""

import json
import os

import pytest

from corread import hub, paths
from corread.hub import _cards, _client, _wordsets


class FakeApi:
    """Stands in for ``huggingface_hub.HfApi``: no repo exists."""

    def repo_exists(self, repo_id, repo_type=None):
        return False


def test_hub_needs_an_organisation(root):
    with pytest.raises(ValueError, match=r'\[hub\]'):
        _client.settings()
    pytest.importorskip('huggingface_hub')
    with pytest.raises(ValueError, match=r'\[hub\]'):
        hub.list_models()


def test_settings(root):
    with open(os.path.join(root, 'paths.toml'), 'a') as f:
        f.write('\n[hub]\norg = "my-lab"\n')
    paths.clear_cache()
    assert _client.settings() == ('my-lab', True)


@pytest.mark.parametrize('path', ['fonts/a.ttf', 'wordlists/english.txt'])
def test_recipe_files_accepted(path):
    assert _wordsets._check_asset_path(path) == path


@pytest.mark.parametrize(
    'path',
    [
        '../a.ttf',
        'fonts/../../a.ttf',
        'fonts/sub/a.ttf',
        '/etc/passwd',
        'fonts/.ttf',
        'fonts/a.py',
        'wordlists\\..\\a.txt',
        'models/a.pth.tar',
    ],
)
def test_recipe_files_refused(path):
    with pytest.raises(ValueError, match='Unexpected file'):
        _wordsets._check_asset_path(path)


@pytest.mark.parametrize(('model', 'ok'), [('pre/pre_z_49', True), ('a/b/c', False), ('../x', False), ('x', False)])
def test_model_names(root, model, ok):
    if ok:
        with pytest.raises(FileNotFoundError):
            hub.export_model(model)
    else:
        with pytest.raises(ValueError):
            hub.export_model(model)


def test_model_card(in_trained):
    pytest.importorskip('huggingface_hub')
    folder = os.path.join(in_trained, 'models', 'words', 'lit_bias_z_3')
    meta = json.load(open(os.path.join(folder, 'metadata.json')))
    card = _cards.model_card(FakeApi(), 'my-lab', 'words', meta)
    assert card.startswith('---\n')
    assert 'corread:' in card and 'my-lab/words.lit_bias_z_3' in card
    assert 'from corread.models import CORnetZBiasedWords' in card
    assert 'weights_only=True' in card
    assert 'net = CORnetZBiasedWords(out_img=1000, out_wrd=1000)' in card
