import os

import pytest

from corread import paths


def test_folders_resolve_relative_to_the_config(root):
    assert paths.get_path('fonts') == os.path.join(root, 'fonts')
    assert paths.find_config() == os.path.join(root, 'paths.toml')


def test_unknown_folder_and_section(root):
    with pytest.raises(KeyError, match='nonexistent'):
        paths.get_path('nonexistent')
    with pytest.raises(KeyError, match=r'\[hub\]'):
        paths.get_section('hub')


def test_found_in_a_parent_of_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / 'paths.toml').write_text('models = "m"\n[report]\nevery = 5\n')
    (tmp_path / 'a' / 'b').mkdir(parents=True)
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path / 'a' / 'b')
    paths.clear_cache()
    try:
        assert paths.get_path('models') == str(tmp_path / 'm')
        assert paths.get_section('report') == {'every': 5}
    finally:
        paths.clear_cache()


def test_missing_config(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / 'missing.toml'))
    paths.clear_cache()
    with pytest.raises(FileNotFoundError, match=paths.ENV_VAR):
        paths.get_path('models')
    paths.clear_cache()
