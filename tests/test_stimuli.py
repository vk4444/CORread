import io
import json
import os

import pytest

import corread
from corread import stimuli

SET = dict(train_fonts=['sans'], val_fonts=['sans'], num_words=4, num_train=3, num_val=2, width=120, height=120, seed=5)


def png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_generate_image_is_deterministic(root):
    a = corread.generate_image('house', 'sans', size=50, xshift=-10, upper=True)
    b = corread.generate_image('house', 'sans', size=50, xshift=-10, upper=True)
    assert a.size == (500, 500) and a.mode == 'RGB'
    assert png(a) == png(b)
    assert png(a) != png(corread.generate_image('house', 'sans', size=50))


def test_generate_training_set(root):
    folder = corread.generate_training_set('ws', 'words', **SET)
    meta = json.load(open(os.path.join(folder, 'metadata.json')))
    assert meta['status'] == 'complete'
    assert meta['words'] == ['cat', 'house', 'tree', 'moon']
    assert meta['seed_used'] == 5
    # the recipe names its input files, not where they are on this machine
    assert meta['wordlist_file']['file'] == 'words.txt' and 'path' not in meta['wordlist_file']
    for split, n in (('train', 3), ('val', 2)):
        for word in meta['words']:
            assert len(os.listdir(os.path.join(folder, split, word))) == n
    checksums = json.load(open(os.path.join(folder, 'checksums.json')))
    assert checksums['images'] == 4 * 5


def test_result_does_not_depend_on_the_workers(root):
    one = corread.generate_training_set('one', 'words', workers=1, **SET)
    two = corread.generate_training_set('two', 'words', workers=2, **SET)
    for name in ('checksums.json', 'images.csv'):
        assert open(os.path.join(one, name), 'rb').read() == open(os.path.join(two, name), 'rb').read()


def test_regenerate_verifies_against_the_recipe(root, tmp_path):
    folder = corread.generate_training_set('ws', 'words', **SET)
    again = corread.regenerate_training_set('ws', recipe=folder, wordsetsdir=str(tmp_path))
    regeneration = json.load(open(os.path.join(again, 'regeneration.json')))
    assert regeneration['verified'] is True
    # the recipe is copied unchanged, so trained models can identify the set
    assert (
        open(os.path.join(again, 'metadata.json'), 'rb').read()
        == open(os.path.join(folder, 'metadata.json'), 'rb').read()
    )


def test_regenerate_detects_different_images(root, tmp_path):
    folder = corread.generate_training_set('ws', 'words', **SET)
    path = os.path.join(folder, 'checksums.json')
    checksums = json.load(open(path))
    checksums['folders']['train/cat'] = '0' * 64
    json.dump(checksums, open(path, 'w'))
    with pytest.raises(RuntimeError, match='differ from the recipe'):
        corread.regenerate_training_set('ws', recipe=folder, wordsetsdir=str(tmp_path))
    assert not os.path.exists(os.path.join(tmp_path, 'ws'))
    kept = corread.regenerate_training_set('ws', recipe=folder, wordsetsdir=str(tmp_path), allow_mismatch=True)
    regeneration = json.load(open(os.path.join(kept, 'regeneration.json')))
    assert regeneration['verified'] is False and regeneration['mismatched_folders'] == ['train/cat']


def test_regenerate_needs_the_same_fonts(root, tmp_path):
    folder = corread.generate_training_set('ws', 'words', **SET)
    with open(os.path.join(root, 'fonts', 'sans.ttf'), 'ab') as f:
        f.write(b'\0')
    with pytest.raises(FileNotFoundError, match='sha256 differs'):
        corread.regenerate_training_set('ws', recipe=folder, wordsetsdir=str(tmp_path))


@pytest.mark.parametrize('name', ['', '.', '..', 'a/b', 'a\\b'])
def test_names_must_be_plain_folder_names(root, name):
    with pytest.raises(ValueError, match='plain folder name'):
        corread.generate_training_set(name, 'words', **SET)


def test_words_must_be_plain_folder_names(root):
    with open(os.path.join(root, 'wordlists', 'bad.txt'), 'w') as f:
        f.write('fine\n../escape\n')
    with pytest.raises(ValueError, match='plain folder name'):
        corread.generate_training_set('ws', 'bad', **SET)


def test_existing_set_is_never_overwritten(root):
    corread.generate_training_set('ws', 'words', **SET)
    with pytest.raises(FileExistsError):
        corread.generate_training_set('ws', 'words', **SET)


def test_plan_is_reproducible():
    a = stimuli._plan_images(['a', 'b'], 1, 3, 2, ['x', 'y'], ['z'], [10, 20], [0, 5], [0], [0, 1])
    b = stimuli._plan_images(['a', 'b'], 1, 3, 2, ['x', 'y'], ['z'], [10, 20], [0, 5], [0], [0, 1])
    assert a == b
    assert all(font == 'z' for _, items in a for split, _, font, *_ in items if split == 'val')
