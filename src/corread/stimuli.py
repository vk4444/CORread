# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""Word images: single stimuli and reproducible training sets.

A word set is a folder of word images, ``train/<word>/`` and ``val/<word>/``, drawn in given fonts
with random size, case and position. Every random choice is drawn up front from one seed, so a set is
fully determined by its *recipe* — the seed, the settings, the wordlist and the font files — and can
be drawn again byte for byte (:func:`regenerate_training_set`). The recipe is written next to the
images:

- ``metadata.json``: arguments, seed, rendering settings, words, and the sha256 of every input file;
- ``images.csv``: the font, size, case and shift drawn for every image;
- ``checksums.json``: the sha256 digest of every word folder.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import logging
import os
import platform
import random
import secrets
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Any

import PIL
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from . import reporting
from ._io import check_name, now_iso, read_json, rename_dir, write_json
from ._provenance import code_provenance, sha256_file
from .paths import get_path

logger = logging.getLogger(__name__)

IMAGES_HEADER = ['file', 'split', 'word', 'index', 'font', 'size', 'uppercase', 'xshift', 'yshift']

FONT_SIZES = [40, 50, 60, 70, 80]
XSHIFTS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
YSHIFTS = [-30, -15, 0, 15, 30]
UPPERCASE = [0, 1]

# A plan lists every image of a set before any is drawn:
# [(word, [(split, index, font, size, uppercase, xshift, yshift), ...]), ...]
Plan = list[tuple[str, list[tuple[str, int, str, int, int, int, int]]]]
Entry = tuple[str, str, str, str]  # (split, word, file name, sha256)


@cache
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    # each font file is loaded once per size and process; a loaded font renders identically every time
    return ImageFont.truetype(path, size)


def _font_path(fontsdir: str, fontname: str) -> str:
    return os.path.join(fontsdir, fontname + '.ttf')


def _draw(
    text: str,
    mirror: bool,
    invert: bool,
    fontname: str,
    fontsdir: str,
    width: int,
    height: int,
    size: int,
    xshift: int,
    yshift: int,
    upper: bool,
) -> tuple[Image.Image, str]:
    """Draw text in black on a white canvas; returns the image and the text as drawn."""
    if upper:
        text = text.upper()
    if invert:
        text = text[::-1]
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    font = _load_font(_font_path(fontsdir, fontname), size)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xshift + (width - (right - left)) / 2, yshift + (height - (bottom - top)) / 2), text, font=font, fill='black'
    )
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img, text


def generate_image(
    text: str,
    fontname: str,
    *,
    fontsdir: str | None = None,
    width: int = 500,
    height: int = 500,
    size: int = 24,
    xshift: int = 0,
    yshift: int = 0,
    upper: bool = False,
    mirror: bool = False,
    invert: bool = False,
) -> Image.Image:
    """Draw one word image, as the word sets do.

    Parameters
    ----------
    text : str
        The word.
    fontname : str
        Font file name without ``.ttf``, in ``fontsdir``.
    fontsdir : str, optional
        Folder of the fonts; by default ``fonts`` from ``paths.toml``.
    width, height : int
        Canvas size in pixels.
    size : int
        Font size.
    xshift, yshift : int
        Offset of the text from the centre of the canvas, in pixels.
    upper, mirror, invert : bool
        Draw the text uppercased, mirrored left to right, or with its letters reversed.

    Returns
    -------
    PIL.Image.Image
        The RGB image.
    """
    if fontsdir is None:
        fontsdir = get_path('fonts')
    return _draw(text, mirror, invert, fontname, fontsdir, width, height, size, xshift, yshift, upper)[0]


def _render_word(task: tuple[str, list, str, str, int, int]) -> list[Entry]:
    """Draw and save every image of one word; module-level so worker processes can run it.

    Each JPEG is encoded in memory, so its hash comes free; the bytes are those ``img.save(path)``
    would write.
    """
    word, items, path_out, fontsdir, width, height = task
    done = []
    for split, n, fontname, size, upper, xshift, yshift in items:
        img, shown = _draw(word, False, False, fontname, fontsdir, width, height, size, xshift, yshift, upper)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        data = buf.getvalue()
        file_name = shown + str(n) + '.jpg'
        with open(os.path.join(path_out, split, word, file_name), 'wb') as f:
            f.write(data)
        done.append((split, word, file_name, hashlib.sha256(data).hexdigest()))
    return done


def _read_words(path: str) -> list[str]:
    """The non-empty lines of a wordlist; each word becomes a folder name, so it must be a plain one."""
    with open(path, encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    for word in words:
        check_name(word, 'every word of ' + path)
    return words


def _plan_images(
    words: list[str],
    seed: int,
    num_train: int,
    num_val: int,
    train_fonts: list[str],
    val_fonts: list[str],
    sizes: list[int],
    xshift_values: list[int],
    yshift_values: list[int],
    uppercase_values: list[int],
) -> Plan:
    """Every image's settings, drawn in a fixed order, so a set depends only on its seed, not on the workers."""
    rng = random.Random(seed)
    plan = []
    for word in words:
        items = []
        for n in range(num_train + num_val):
            if n < num_train:
                split, fontname = 'train', rng.choice(train_fonts)
            else:
                split, fontname = 'val', rng.choice(val_fonts)
            size = rng.choice(sizes)
            upper = rng.choice(uppercase_values)
            xshift = rng.choice(xshift_values)
            yshift = rng.choice(yshift_values)
            items.append((split, n, fontname, size, upper, xshift, yshift))
        plan.append((word, items))
    return plan


def _image_rows(plan: Plan):
    """The rows of ``images.csv`` (after its header) for a plan."""
    for word, items in plan:
        for split, n, fontname, size, upper, xshift, yshift in items:
            shown = word.upper() if upper else word
            yield [
                split + '/' + word + '/' + shown + str(n) + '.jpg',
                split,
                word,
                n,
                fontname,
                size,
                upper,
                xshift,
                yshift,
            ]


def _make_folders(path_out: str, words: list[str]) -> None:
    os.makedirs(path_out)
    for split in ('train', 'val'):
        for word in words:
            os.makedirs(os.path.join(path_out, split, word), exist_ok=True)


def _render(
    plan: Plan,
    path_out: str,
    fontsdir: str,
    width: int,
    height: int,
    workers: int,
    on_word: Callable[[], None] | None = None,
    desc: str = 'Generating word images',
) -> list[Entry]:
    """Draw a plan's images into ``path_out/<split>/<word>/``, in worker processes when ``workers > 1``."""
    tasks = [(word, items, path_out, fontsdir, width, height) for word, items in plan]
    entries: list[Entry] = []

    def collect(results):
        for done in tqdm(results, total=len(tasks), desc=desc, unit='word'):
            entries.extend(done)
            if on_word is not None:
                on_word()

    if workers == 1:
        collect(map(_render_word, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            collect(executor.map(_render_word, tasks))
    return entries


def _digest(pairs) -> str:
    h = hashlib.sha256()
    for a, b in sorted(pairs):
        h.update((a + ' ' + b + '\n').encode('utf-8'))
    return h.hexdigest()


def _checksums(entries: list[Entry]) -> dict[str, Any]:
    """The contents of ``checksums.json``: a digest per word folder, and one over the whole set."""
    folders: dict[str, list] = {}
    for split, word, file_name, sha in entries:
        folders.setdefault(split + '/' + word, []).append((file_name, sha))
    digests = {folder: _digest(files) for folder, files in folders.items()}
    return {
        'algorithm': 'sha256',
        'folder_digest': "sha256 of the lines '<file name> <sha256 of the file>\\n' of the folder's images, "
        'sorted by file name',
        'total_digest': "sha256 of the lines '<split>/<word> <folder digest>\\n', sorted",
        'images': sum(len(files) for files in folders.values()),
        'total': _digest(digests.items()),
        'folders': dict(sorted(digests.items())),
    }


def _recipe_inputs(meta: dict, fontsdir: str, wordsdir: str) -> tuple[dict[str, str], str]:
    """The font files and wordlist a recipe needs, checked against the sha256 it recorded.

    Returns ``({font name: path}, wordlist path)``.
    """
    problems = []

    def check(path, sha):
        if not os.path.isfile(path):
            problems.append(path + ' is missing')
        elif sha256_file(path) != sha:
            problems.append(path + ' is not the file the set was made with (sha256 differs)')

    words_path = os.path.join(wordsdir, meta['arguments']['wordlist'] + '.txt')
    check(words_path, (meta.get('wordlist_file') or {}).get('sha256'))
    fonts = {}
    for fontname, recorded in sorted((meta.get('font_files') or {}).items()):
        fonts[fontname] = _font_path(fontsdir, fontname)
        check(fonts[fontname], recorded.get('sha256'))
    if problems:
        raise FileNotFoundError(
            'Word set ' + str(meta.get('name')) + ' needs its own fonts and wordlist: ' + '; '.join(problems)
        )
    return fonts, words_path


def _check_draws(plan: Plan, images_csv: str) -> None:
    """Raise at the first image whose drawn settings differ from ``images.csv``."""
    with open(images_csv, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != IMAGES_HEADER:
            raise ValueError(images_csv + ' does not start with the header ' + ','.join(IMAGES_HEADER))
        for line, (drawn, recorded) in enumerate(itertools.zip_longest(_image_rows(plan), reader), start=2):
            drawn = None if drawn is None else [str(v) for v in drawn]
            if drawn != recorded:
                raise RuntimeError(
                    'The random draws differ from '
                    + images_csv
                    + ' at line '
                    + str(line)
                    + ': drawn '
                    + str(drawn)
                    + ', recorded '
                    + str(recorded)
                )


def _generator(function: str) -> dict[str, Any]:
    return {
        'function': 'corread.stimuli.' + function,
        **code_provenance(),
        'python': sys.version.split()[0],
        'pillow': PIL.__version__,
        'platform': platform.platform(),
    }


def generate_training_set(
    name: str,
    wordlist: str,
    *,
    train_fonts: list[str],
    val_fonts: list[str],
    num_words: int = 1000,
    num_train: int = 1000,
    num_val: int = 50,
    fontsdir: str | None = None,
    wordsdir: str | None = None,
    wordsetsdir: str | None = None,
    width: int = 500,
    height: int = 500,
    seed: int | None = None,
    workers: int = 1,
    report: bool = False,
    export: bool = False,
) -> str:
    """Generate a word set: ``num_train`` training and ``num_val`` validation images of every word.

    The training images use ``train_fonts`` and the validation images ``val_fonts``, so validation can
    test generalization to an unseen font. Each image draws its font, size (40–80), case, and a shift of
    up to ±50 px horizontally and ±30 px vertically.

    Parameters
    ----------
    name : str
        Name of the set: its folder in ``wordsetsdir``.
    wordlist : str
        Wordlist file name without ``.txt``, in ``wordsdir``: one word per line.
    train_fonts, val_fonts : list of str
        Font file names without ``.ttf``, in ``fontsdir``.
    num_words : int
        Use the first ``num_words`` words of the wordlist.
    num_train, num_val : int
        Images per word in each split.
    fontsdir, wordsdir, wordsetsdir : str, optional
        Folders of the fonts, the wordlists and the word sets; by default from ``paths.toml``.
    width, height : int
        Canvas size in pixels.
    seed : int, optional
        Seed of every random choice; a random one is drawn and recorded when not given.
    workers : int
        Processes drawing images; the result does not depend on it.
    report : bool
        Send the progress to the reporting API set in ``paths.toml`` (see ``docs/reporting.md``).
    export : bool
        Upload the set's recipe to the Hugging Face Hub when it is complete (see :mod:`corread.hub`).

    Returns
    -------
    str
        The folder of the new set.
    """
    started = time.time()
    wordsetsdir = wordsetsdir if wordsetsdir is not None else get_path('wordsets')
    fontsdir = fontsdir if fontsdir is not None else get_path('fonts')
    wordsdir = wordsdir if wordsdir is not None else get_path('wordlists')
    if workers < 1:
        raise ValueError('workers must be at least 1.')
    # fail now on a missing setting, before any work
    settings = reporting.settings() if report else None
    if export:
        from .hub import connect

        connect(write=True)
    check_name(name)
    path_out = os.path.join(wordsetsdir, name)
    if os.path.exists(path_out):
        raise FileExistsError('Word set already exists: ' + path_out + ' (delete it or choose another name)')
    if not train_fonts:
        raise ValueError('You must provide at least one training font name.')
    if not val_fonts:
        raise ValueError('You must provide at least one validation font name.')
    for fontname in set(train_fonts) | set(val_fonts):
        if not os.path.isfile(_font_path(fontsdir, fontname)):
            raise FileNotFoundError('Font not found: ' + _font_path(fontsdir, fontname))
    words_path = os.path.join(wordsdir, wordlist + '.txt')
    if not os.path.isfile(words_path):
        raise FileNotFoundError('Wordlist not found: ' + words_path)

    seed_used = seed if seed is not None else secrets.randbits(64)
    all_words = _read_words(words_path)
    words = all_words[:num_words]
    _make_folders(path_out, words)
    plan = _plan_images(
        words, seed_used, num_train, num_val, train_fonts, val_fonts, FONT_SIZES, XSHIFTS, YSHIFTS, UPPERCASE
    )

    meta_path = os.path.join(path_out, 'metadata.json')
    meta = {
        'name': name,
        'run_id': str(uuid.uuid4()),
        'status': 'in progress',
        'started_at': now_iso(started),
        'finished_at': None,
        'duration_seconds': None,
        'arguments': {
            'name': name,
            'wordlist': wordlist,
            'num_words': num_words,
            'num_train': num_train,
            'num_val': num_val,
            'train_fonts': list(train_fonts),
            'val_fonts': list(val_fonts),
            'width': width,
            'height': height,
            'seed': seed,
            'workers': workers,
        },
        'seed_used': seed_used,
        'reproduce': 'corread.stimuli.regenerate_training_set draws the images again from this file, the wordlist '
        'and the font files (checked by the hashes below), and checks them against checksums.json',
        'generator': _generator('generate_training_set'),
        'wordlist_file': {
            'file': wordlist + '.txt',
            'sha256': sha256_file(words_path),
            'words_in_file': len(all_words),
            'words_used': len(words),
            'selection': 'first num_words non-empty lines, in file order',
        },
        'font_files': {
            fontname: {'file': fontname + '.ttf', 'sha256': sha256_file(_font_path(fontsdir, fontname))}
            for fontname in sorted(set(train_fonts) | set(val_fonts))
        },
        'rendering': {
            'canvas': {'mode': 'RGB', 'width': width, 'height': height, 'background': [255, 255, 255]},
            'text_color': 'black',
            'font_sizes': FONT_SIZES,
            'xshift_values': XSHIFTS,
            'yshift_values': YSHIFTS,
            'uppercase_values': UPPERCASE,
            'mirror': False,
            'invert': False,
            'placement': 'text bounding box centred on the canvas, then moved by (xshift, yshift) pixels',
            'random_draws': 'random.Random(seed_used); for each word in list order, for index 0 .. '
            'num_train+num_val-1, one uniform choice each of: font (train fonts for '
            'index < num_train, else val fonts), size, uppercase, xshift, yshift',
            'file_format': 'JPEG, Pillow default settings (quality 75, 4:2:0 chroma subsampling)',
            'file_name': '<split>/<word>/<word, uppercased if uppercase=1><index>.jpg',
        },
        'splits': {
            'train': {
                'indices': [0, num_train - 1],
                'images_per_word': num_train,
                'total_images': num_train * len(words),
                'fonts': list(train_fonts),
            },
            'val': {
                'indices': [num_train, num_train + num_val - 1],
                'images_per_word': num_val,
                'total_images': num_val * len(words),
                'fonts': list(val_fonts),
            },
        },
        'words': words,
        'class_to_idx': {word: i for i, word in enumerate(sorted(words))},
        'class_to_idx_note': 'the index torchvision ImageFolder assigns (sorted folder names); training adds '
        'img_classes to it',
        'images_file': 'images.csv',
        'checksums_file': 'checksums.json',
    }
    write_json(meta_path, meta)

    reporter = reporting.Reporter(*settings) if settings else None
    words_done = 0

    def send(event, status, error=None):
        # never raises: a status report must not stop the generation
        if reporter is None:
            return
        try:
            reporter.send(reporting.wordset_payload(meta, event, status, words_done, error))
        except Exception as e:
            logger.warning('could not build the status report (%s)', e)

    def word_done():
        nonlocal words_done
        words_done += 1
        if reporter is not None and reporter.due():
            send('progress', 'in progress')

    send('started', 'in progress')
    try:
        with open(os.path.join(path_out, 'images.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(IMAGES_HEADER)
            writer.writerows(_image_rows(plan))
        entries = _render(plan, path_out, fontsdir, width, height, workers, on_word=word_done)
        write_json(os.path.join(path_out, 'checksums.json'), _checksums(entries))
        finished = time.time()
        meta.update(status='complete', finished_at=now_iso(finished), duration_seconds=round(finished - started, 1))
        write_json(meta_path, meta)
        send('complete', 'complete')
    except KeyboardInterrupt:
        send('interrupted', 'interrupted')
        raise
    except Exception as e:
        send('failed', 'failed', error=e)
        raise
    finally:
        if reporter is not None:
            reporter.close()

    if export:
        from .hub import export_wordset_folder

        export_wordset_folder(path_out, fontsdir=fontsdir, wordsdir=wordsdir)
    return path_out


def regenerate_training_set(
    name: str,
    recipe: str | None = None,
    *,
    workers: int = 1,
    wordsetsdir: str | None = None,
    fontsdir: str | None = None,
    wordsdir: str | None = None,
    allow_mismatch: bool = False,
) -> str:
    """Draw a word set again from its recipe and check that the images are identical.

    The draws are checked against ``images.csv`` before any image is drawn, and the images against
    ``checksums.json`` after. Text rendering and JPEG encoding come from the FreeType and libjpeg that
    Pillow bundles, so identical images need the Pillow version recorded in the recipe.

    Parameters
    ----------
    name : str
        Name of the set; it is drawn into ``wordsetsdir/name``, which must not exist.
    recipe : str, optional
        Folder with the recipe files (``metadata.json``, ``images.csv``, ``checksums.json``); an existing
        set's folder works too. By default ``wordsetsdir/<name>.recipe``, where
        :func:`corread.hub.import_wordset` puts them.
    workers : int
        Processes drawing images.
    wordsetsdir, fontsdir, wordsdir : str, optional
        By default from ``paths.toml``. The fonts and wordlist must be the files the set was made with.
    allow_mismatch : bool
        Keep images that differ from the recipe (recorded as ``verified: false`` in
        ``regeneration.json``) instead of raising.

    Returns
    -------
    str
        The folder of the regenerated set.
    """
    started = time.time()
    wordsetsdir = wordsetsdir if wordsetsdir is not None else get_path('wordsets')
    fontsdir = fontsdir if fontsdir is not None else get_path('fonts')
    wordsdir = wordsdir if wordsdir is not None else get_path('wordlists')
    recipe = recipe if recipe is not None else os.path.join(wordsetsdir, name + '.recipe')
    if workers < 1:
        raise ValueError('workers must be at least 1.')
    check_name(name)
    path_out = os.path.join(wordsetsdir, name)
    if os.path.exists(path_out):
        raise FileExistsError('Word set already exists: ' + path_out + ' (delete it or rename it first)')

    meta_path = os.path.join(recipe, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError('No word set recipe: ' + meta_path + ' does not exist')
    meta = read_json(meta_path)
    if meta.get('status') != 'complete':
        raise RuntimeError(meta_path + ' is not of a complete word set (status: ' + str(meta.get('status')) + ')')
    if meta.get('name') != name:
        raise ValueError(meta_path + ' is the recipe of ' + repr(meta.get('name')) + ', not ' + repr(name))

    args, rendering = meta['arguments'], meta['rendering']
    _, words_path = _recipe_inputs(meta, fontsdir, wordsdir)
    words = _read_words(words_path)[: args['num_words']]
    if words != meta['words']:
        raise RuntimeError(words_path + ' does not give the words in ' + meta_path)
    recorded = meta.get('generator') or {}
    if recorded.get('code_sha256') != code_provenance()['code_sha256']:
        logger.info(
            'the code differs from the version that made %s; the checksums show whether the images still match', name
        )

    # the recipe's own rendering values, so later changes to the defaults don't change what it regenerates
    plan = _plan_images(
        words,
        meta['seed_used'],
        args['num_train'],
        args['num_val'],
        args['train_fonts'],
        args['val_fonts'],
        rendering['font_sizes'],
        rendering['xshift_values'],
        rendering['yshift_values'],
        rendering['uppercase_values'],
    )
    images_csv = os.path.join(recipe, 'images.csv')
    if os.path.isfile(images_csv):
        _check_draws(plan, images_csv)
    checksums_path = os.path.join(recipe, 'checksums.json')
    expected = read_json(checksums_path) if os.path.isfile(checksums_path) else None

    # drawn into <name>.part/ and renamed when complete and checked, so a word set folder is always whole
    part = path_out + '.part'
    shutil.rmtree(part, ignore_errors=True)
    try:
        _make_folders(part, words)
        entries = _render(plan, part, fontsdir, args['width'], args['height'], workers, desc='Regenerating ' + name)
        got = _checksums(entries)
        mismatched = (
            []
            if expected is None
            else sorted(
                folder
                for folder in set(expected['folders']) | set(got['folders'])
                if expected['folders'].get(folder) != got['folders'].get(folder)
            )
        )
        if mismatched and not allow_mismatch:
            raise RuntimeError(
                'The regenerated images differ from the recipe in '
                + str(len(mismatched))
                + ' of '
                + str(len(expected['folders']))
                + ' word folders (e.g. '
                + ', '.join(mismatched[:5])
                + '). '
                + name
                + ' was made with Pillow '
                + str(recorded.get('pillow'))
                + '; this is Pillow '
                + PIL.__version__
                + ', and text rendering and JPEG encoding depend on the Pillow build. '
                'Install Pillow ' + str(recorded.get('pillow')) + ', or pass allow_mismatch=True to keep '
                'these images (recorded as verified: false)'
            )
        # the recipe's files, unchanged, so the metadata_sha256 trained models record matches the original's
        for file_name in sorted(os.listdir(recipe)):
            src = os.path.join(recipe, file_name)
            if os.path.isfile(src) and file_name not in ('README.md', '.gitattributes', 'regeneration.json'):
                shutil.copyfile(src, os.path.join(part, file_name))
        if expected is None:
            write_json(os.path.join(part, 'checksums.json'), got)
        write_json(
            os.path.join(part, 'regeneration.json'),
            {
                'regenerated_at': now_iso(),
                'verified': None if expected is None else not mismatched,
                'checked_against': None if expected is None else "the recipe's checksums.json",
                'mismatched_folders': mismatched[:100],
                'mismatched_count': len(mismatched),
                'images': got['images'],
                'images_digest': got['total'],
                'pillow': PIL.__version__,
                'recipe_pillow': recorded.get('pillow'),
                'generator': _generator('regenerate_training_set'),
                'workers': workers,
                'duration_seconds': round(time.time() - started, 1),
            },
        )
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    rename_dir(part, path_out)
    if expected is None:
        logger.info('Regenerated %s (not verified: the recipe has no checksums.json)', path_out)
    elif mismatched:
        logger.warning(
            'Regenerated %s; %d word folders differ from the recipe (kept: allow_mismatch=True)',
            path_out,
            len(mismatched),
        )
    else:
        logger.info('Regenerated %s: all %d images match the recipe', path_out, got['images'])
    return path_out


def report_wordset_status(name: str, wordsetsdir: str | None = None) -> None:
    """Send the current status of word set ``name`` to the reporting API now; raises if it fails."""
    wordsetsdir = wordsetsdir if wordsetsdir is not None else get_path('wordsets')
    url, token, _ = reporting.settings()
    meta_path = os.path.join(wordsetsdir, name, 'metadata.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError('No word set to report: ' + meta_path + ' does not exist')
    meta = read_json(meta_path)
    reporting.post(url, token, reporting.wordset_payload(meta, 'snapshot', meta.get('status')))
